use std::{alloc::Layout, collections::HashSet, mem::MaybeUninit, rc::Rc, slice};

use gpurs_driver::{LaunchAttributes, Module};

use crate::{
    Argument, Buffer, DeviceAddress, DeviceSlice, DeviceSliceMut, Error, ExternalAccess, Gpu,
    Launch, LaunchConfig, PinnedBuffer, Result, Stream, Transfer,
    gpu::Scatter,
    host::{
        graph::{CapturedKernel, CapturedOperation},
        tracking::{self, Access},
    },
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Duplicates {
    #[default]
    Disable,
    KeepLast,
}

pub struct DeviceScatter<T> {
    buffer: Buffer<Scatter<T>>,
    dependencies: [ExternalAccess; 2],
    indices: Buffer<usize>,
    target_len: usize,
    duplicates: Duplicates,
    validation: Option<(Buffer<usize>, Buffer<u64>)>,
    _upload: Option<Transfer<PinnedBuffer<Scatter<T>>>>,
}

impl<T: Copy> DeviceScatter<T> {
    /// Snapshots GPU indices asynchronously, suppressing out-of-bounds entries and resolving
    /// duplicate destinations by `duplicates`, so every destination keeps at most one entry.
    /// Subsequent source changes do not affect this view; use `DeviceScatter::update_device`
    /// to refresh it. A capturing `queue` records the validation into its graph.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn from_device(
        queue: &impl Launch,
        target: &mut impl DeviceSliceMut<T>,
        indices: &impl DeviceSlice<usize>,
        duplicates: Duplicates,
    ) -> Result<DeviceScatter<T>> {
        let stream = queue.stream();
        let gpu = Gpu {
            device: stream.device.clone(),
        };
        let span = target.span();
        if span.len == 0
            || indices.is_empty()
            || !Rc::ptr_eq(&stream.device, &span.allocation.device)
            || !Rc::ptr_eq(&stream.device, &stream.device)
        {
            return Err(Error::InvalidOperation(
                "scatter views require nonempty ranges on the same GPU".to_owned(),
            ));
        }
        let allocation = gpu.allocate(Layout::array::<usize>(indices.len()).map_err(|_| {
            Error::InvalidOperation("scatter index allocation exceeds the size range".to_owned())
        })?)?;
        let target_len = span.len;
        let descriptor = Scatter {
            // SAFETY: the target is retained by the descriptor's tracked dependency.
            target: unsafe { DeviceAddress::from_raw(span.allocation.pointer + span.start as u64) },
            // SAFETY: validation initializes the retained private index allocation.
            indices: unsafe { DeviceAddress::from_raw(allocation.address()) },
            len: indices.len(),
            target_len: span.len,
        };
        let mut snapshot = DeviceScatter {
            buffer: Buffer::new(gpu.allocate(Layout::new::<Scatter<T>>())?.allocation, 1),
            dependencies: [
                ExternalAccess::write(target),
                ExternalAccess(Access::read(&allocation.allocation, 0..allocation.len())),
            ],
            indices: Buffer::new(allocation.allocation, indices.len()),
            target_len,
            duplicates,
            validation: None,
            _upload: None,
        };
        snapshot.update_device(queue, indices)?;
        snapshot._upload =
            Some(stream.try_copy_descriptor(&mut snapshot.buffer, slice::from_ref(&descriptor))?);
        Ok(snapshot)
    }

    /// Creates retained device storage and verifies its construction requirements.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn new(
        stream: &Stream,
        target: &mut impl DeviceSliceMut<T>,
        indices: &[usize],
    ) -> Result<DeviceScatter<T>> {
        let gpu = Gpu {
            device: stream.device.clone(),
        };
        let span = target.span();
        if span.len == 0
            || indices.is_empty()
            || !Rc::ptr_eq(&stream.device, &span.allocation.device)
            || indices.iter().any(|&index| index >= span.len)
            || indices.iter().collect::<HashSet<_>>().len() != indices.len()
        {
            return Err(Error::InvalidOperation(
                "scatter views require nonempty ranges on the same GPU and unique, in-bounds \
                 indices"
                    .to_owned(),
            ));
        }
        let indices = gpu.copy(indices)?;
        unsafe { Self::new_unchecked(stream, target, &indices) }
    }

    /// Retains GPU-resident indices without downloading or validating their contents.
    /// Out-of-range destinations are ignored by stores and return `None` from loads.
    ///
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    /// # Safety
    /// In-bounds destinations must be unique for every launch using this descriptor,
    /// including after changes through aliases of the index buffer. The index range
    /// must not overlap the target range. Index producers must use tracked GPU accesses.
    pub unsafe fn new_unchecked(
        stream: &Stream,
        target: &mut impl DeviceSliceMut<T>,
        indices: &impl DeviceSlice<usize>,
    ) -> Result<DeviceScatter<T>> {
        let gpu = Gpu {
            device: stream.device.clone(),
        };
        let span = target.span();
        let indices_span = indices.span();
        if span.len == 0
            || indices_span.len == 0
            || !Rc::ptr_eq(&stream.device, &span.allocation.device)
            || !Rc::ptr_eq(&stream.device, &indices_span.allocation.device)
        {
            return Err(Error::InvalidOperation(
                "scatter views require nonempty ranges on the same GPU".to_owned(),
            ));
        }
        let mut indices = Buffer::new(indices_span.allocation.clone(), indices_span.len);
        indices.offset = indices_span.start;
        let target_len = span.len;
        let descriptor = Scatter {
            target: unsafe { DeviceAddress::from_raw(span.allocation.pointer + span.start as u64) },
            indices: indices.address(),
            len: indices.len(),
            target_len,
        };
        Ok(DeviceScatter {
            buffer: gpu.copy(slice::from_ref(&descriptor))?,
            dependencies: [
                ExternalAccess::write(target),
                ExternalAccess::read(&indices),
            ],
            indices,
            target_len,
            duplicates: Duplicates::Disable,
            validation: None,
            _upload: None,
        })
    }
}

impl<T> DeviceScatter<T> {
    /// Revalidates a same-length GPU index buffer without host readback or synchronization.
    /// Out-of-bounds indices disable their entries and duplicate destinations resolve by the
    /// policy the view was created with.
    /// Existing graph consumers automatically wait for the refreshed snapshot.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn update_device(
        &mut self,
        queue: &impl Launch,
        indices: &impl DeviceSlice<usize>,
    ) -> Result<()> {
        let stream = queue.stream();
        if indices.len() != self.len() {
            return Err(Error::InvalidOperation(
                "scatter update lengths differ".to_owned(),
            ));
        }
        let blocks = u32::try_from(self.len().div_ceil(256)).map_err(|_| {
            Error::InvalidOperation("scatter validation grid exceeds CUDA range".to_owned())
        })?;
        let source = indices.span();
        let device = &self.indices.allocation.device;
        if !Rc::ptr_eq(device, &stream.device) || !Rc::ptr_eq(device, &source.allocation.device) {
            return Err(Error::InvalidOperation(
                "scatter update contexts differ".to_owned(),
            ));
        }
        let _context = device.context.try_enter()?;
        if device.scatter_validation.get().is_none() {
            let _ = device
                .scatter_validation
                .set(Module::try_load_ptx(include_str!("scatter.ptx"))?);
        }
        if self.validation.is_none() {
            let gpu = Gpu {
                device: device.clone(),
            };
            let snapshot = gpu.allocate(Layout::array::<usize>(self.len()).map_err(|_| {
                Error::InvalidOperation(
                    "scatter index allocation exceeds the size range".to_owned(),
                )
            })?)?;
            let allocation =
                gpu.allocate(Layout::array::<u64>(self.target_len).map_err(|_| {
                    Error::InvalidOperation(
                        "scatter ownership allocation exceeds the size range".to_owned(),
                    )
                })?)?;
            self.validation = Some((
                Buffer::new(snapshot.allocation, self.len()),
                Buffer::new(allocation.allocation, self.target_len),
            ));
        }
        let (snapshot, owners) = self
            .validation
            .as_ref()
            .expect("scatter validation storage was allocated");
        let accesses = [
            Access::read(source.allocation, source.bytes::<usize>()),
            Access::write(
                &self.indices.allocation,
                self.indices.offset..self.indices.offset + self.len() * size_of::<usize>(),
            ),
            Access::write(&snapshot.allocation, 0..snapshot.allocation.bytes),
            Access::write(&owners.allocation, 0..owners.allocation.bytes),
        ];
        let arguments = [
            snapshot.allocation.pointer,
            owners.allocation.pointer,
            self.len() as u64,
            self.target_len as u64,
            self.indices.allocation.pointer + self.indices.offset as u64,
            u64::from(self.duplicates == Duplicates::KeepLast),
        ]
        .into_iter()
        .flat_map(u64::to_ne_bytes)
        .map(MaybeUninit::new)
        .collect::<Vec<_>>();
        let mut operations = vec![
            CapturedOperation::copy([accesses[0].clone(), accesses[2].clone()]),
            CapturedOperation::fill(accesses[3].clone(), 0),
        ];
        for name in ["claim_indices", "select_indices"] {
            operations.push(CapturedOperation::kernel(CapturedKernel {
                function: device
                    .scatter_validation
                    .get()
                    .expect("validation module was loaded")
                    .try_function(name)?,
                request: LaunchConfig::new([blocks as usize, 1, 1], [256, 1, 1])
                    .request(LaunchAttributes::default()),
                arguments: arguments.clone(),
                accesses: accesses[1..].to_vec(),
            }));
        }
        if let Some(capture) = queue.capture() {
            capture.borrow_mut().extend(operations);
            return Ok(());
        }
        stream.maintain()?;
        tracking::try_order(&stream.raw, &accesses)?;
        let result = operations
            .iter_mut()
            .try_for_each(|operation| operation.try_submit(&stream.raw));
        tracking::complete(&stream.raw, &accesses);
        result.map_err(Error::from)
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Enqueues validated replacement indices without waiting for prior descriptor reads.
    ///
    /// # Errors
    /// Returns CUDA failures or invalid operation errors.
    pub fn update(
        &mut self,
        stream: &Stream,
        indices: &[usize],
    ) -> Result<Transfer<PinnedBuffer<usize>>> {
        if indices.len() != self.len()
            || indices.iter().any(|&index| index >= self.target_len)
            || indices.iter().collect::<HashSet<_>>().len() != indices.len()
        {
            return Err(Error::InvalidOperation(
                "scatter updates require the same length and unique, in-bounds indices".to_owned(),
            ));
        }
        stream.copy_from_host(&mut self.indices, indices)
    }
}

impl<T> Argument<Scatter<T>> for DeviceScatter<T> {
    fn buffer(&self) -> &Buffer<Scatter<T>> {
        &self.buffer
    }

    fn dependencies(&self) -> &[ExternalAccess] {
        &self.dependencies
    }
}
