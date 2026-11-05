use std::{cell::RefCell, ffi::c_void, mem::MaybeUninit, ptr, rc::Rc};

use gpurs_driver::{self as driver, KernelNodeParameters};

use crate::{
    Error, Event, Result, Stream,
    host::{
        graph::{Capture, CapturedOperation, Graph, GraphExec, Operation},
        tracking,
    },
};

impl GraphExec {
    pub fn upload(&self, stream: &Stream) -> Result<Event> {
        if !Rc::ptr_eq(&self.graph.0.device, &stream.device) {
            return Err(Error::InvalidOperation(
                "graph and stream contexts differ".to_owned(),
            ));
        }
        self.try_finish()?;
        let _context = stream.device.context.try_enter()?;
        self.raw.try_upload(&stream.raw)?;
        let event = match stream.record() {
            Ok(event) => event,
            Err(error) => {
                stream.raw.try_synchronize()?;
                return Err(error);
            }
        };
        *self.pending.borrow_mut() = Some(event.clone());
        *self.graph.0.pending.borrow_mut() = Some(event.clone());
        Ok(event)
    }

    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    pub fn launch(&self, stream: &Stream) -> Result<Event> {
        if !Rc::ptr_eq(&self.graph.0.device, &stream.device) {
            return Err(Error::InvalidOperation(
                "graph and stream contexts differ".to_owned(),
            ));
        }
        stream.maintain()?;
        let _context = stream.device.context.try_enter()?;
        for pending in [&self.graph.0.pending, &self.pending] {
            if let Some(event) = pending.borrow().as_ref() {
                stream.raw.try_wait(&event.raw)?;
            }
        }
        let mut statuses = tracking::try_order(&stream.raw, &self.accesses)?;
        let snapshots = self
            .graph
            .0
            .statuses
            .iter()
            .map(|status| stream.device.status(&status.kernel))
            .collect::<Result<Vec<_>>>()?;
        unsafe { self.raw.try_launch(&stream.raw)? };
        for (source, destination) in self.graph.0.statuses.iter().zip(&snapshots) {
            if let Err(error) = unsafe {
                driver::try_copy_device_async(destination.pointer, source.pointer, 4, &stream.raw)
            } {
                stream.raw.try_synchronize()?;
                return Err(error.into());
            }
        }
        statuses.extend(snapshots);
        tracking::complete(&stream.raw, &self.accesses, &statuses);
        stream.retain_statuses(&statuses);
        let event = match stream.record_statuses(statuses.clone()) {
            Ok(event) => event,
            Err(error) => {
                stream.raw.try_synchronize()?;
                tracking::read_statuses(&statuses)?;
                return Err(error);
            }
        };
        *self.pending.borrow_mut() = Some(event.clone());
        *self.graph.0.pending.borrow_mut() = Some(event.clone());
        Ok(event)
    }

    pub fn update(&mut self, graph: &Graph) -> Result<i32> {
        self.try_finish()?;
        if !Rc::ptr_eq(&self.graph.0.device, &graph.0.device) {
            return Err(Error::InvalidOperation("graph contexts differ".to_owned()));
        }
        let result = self.raw.try_update(&graph.0.raw)?;
        if result == 0 {
            self.graph.clone_from(graph);
            self.replacements = (0..graph.0.nodes.len()).map(|_| None).collect();
            self.accesses.clone_from(&graph.0.accesses);
        }
        Ok(result)
    }

    pub fn update_kernel(
        &mut self,
        index: usize,
        stream: &Stream,
        record: impl FnOnce(&Capture<'_>) -> Result<()>,
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.graph.0.device, &stream.device) {
            return Err(Error::InvalidOperation(
                "graph and stream contexts differ".to_owned(),
            ));
        }
        let operations = self
            .graph
            .0
            ._resources
            .downcast_ref::<Vec<CapturedOperation>>();
        let original = operations
            .and_then(|operations| {
                self.origin.0.nodes.get(index)?;
                operations
                    .iter()
                    .filter_map(|operation| match &operation.0 {
                        Operation::Kernel(kernel) => Some(kernel),
                        _ => None,
                    })
                    .nth(index)
            })
            .ok_or_else(|| Error::InvalidOperation("graph kernel index is invalid".to_owned()))?;
        let kernels = RefCell::new(Vec::new());
        record(&Capture {
            stream,
            config: None,
            kernels: Some(&kernels),
        })?;
        let mut kernels = kernels.into_inner();
        if kernels.len() != 1 {
            return Err(Error::InvalidOperation(
                "graph update requires exactly one kernel".to_owned(),
            ));
        }
        let Operation::Kernel(mut kernel) = kernels.pop().expect("one operation was recorded").0
        else {
            return Err(Error::InvalidOperation(
                "graph update requires a kernel call".to_owned(),
            ));
        };
        if kernel.function.as_raw() != original.function.as_raw() {
            return Err(Error::InvalidOperation(
                "graph update must preserve the kernel entry point".to_owned(),
            ));
        }
        if let Some(status) = &original.status {
            kernel.arguments[..8]
                .copy_from_slice(&status.pointer.to_ne_bytes().map(MaybeUninit::new));
        }
        let mut parameters = [kernel.arguments.as_mut_ptr().cast()];
        let _context = stream.device.context.try_enter()?;
        // SAFETY: the call is verified and its resources and status pointer are retained.
        unsafe {
            self.raw.try_set_kernel(
                self.origin.0.nodes[index],
                &KernelNodeParameters {
                    function: kernel.function.as_raw(),
                    grid: kernel.config.grid.map(|value| value as u32),
                    block: kernel.config.block.map(|value| value as u32),
                    shared_bytes: kernel.config.shared_bytes as u32,
                    parameters: parameters.as_mut_ptr(),
                    extra: ptr::null_mut(),
                },
            )?;
        }
        self.replacements[index] = Some(kernel.accesses);
        let mut kernel_index = 0;
        self.accesses = operations
            .expect("verified graph retains its operations")
            .iter()
            .flat_map(|operation| {
                if matches!(&operation.0, Operation::Kernel(_)) {
                    let accesses = self.replacements[kernel_index]
                        .as_deref()
                        .unwrap_or_else(|| operation.accesses());
                    kernel_index += 1;
                    accesses
                } else {
                    operation.accesses()
                }
            })
            .cloned()
            .collect();
        tracking::deduplicate(&mut self.accesses);
        Ok(())
    }

    /// # Safety
    /// The node, parameter ABI, resources, effects, and status pointers obey `set_kernel`.
    pub unsafe fn set_kernel(
        &mut self,
        node: *mut c_void,
        parameters: &KernelNodeParameters,
    ) -> Result<()> {
        let _context = self.graph.0.device.context.try_enter()?;
        unsafe { self.raw.try_set_kernel(node, parameters)? };
        Ok(())
    }

    fn try_finish(&self) -> Result<()> {
        let _context = self.graph.0.device.context.try_enter()?;
        for pending in [&self.graph.0.pending, &self.pending] {
            if let Some(event) = pending.borrow().as_ref() {
                event.raw.try_synchronize()?;
                event
                    .stream
                    .completed
                    .set(event.stream.completed.get().max(event.sequence));
                tracking::read_statuses(&event.statuses)?;
            }
        }
        let mut result = Ok(());
        for pending in [&self.graph.0.pending, &self.pending] {
            if let Some(event) = pending.take() {
                result = result.and(
                    event
                        .statuses
                        .iter()
                        .filter(|status| !status.reported.get())
                        .try_for_each(|status| status.failure()),
                );
            }
        }
        result
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Some(event) = self.pending.get_mut().take() {
            let _ = event.raw.try_synchronize();
        }
    }
}
