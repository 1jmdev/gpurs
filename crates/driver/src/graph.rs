use std::{ffi::c_void, ptr};

use crate::{Context, Ownership, Result, Stream, ffi};

#[derive(Debug)]
pub struct Graph {
    raw: *mut c_void,
    context: Context,
    ownership: Ownership,
}

#[derive(Debug)]
pub struct GraphExec {
    raw: *mut c_void,
    context: Context,
}

/// CUDA_KERNEL_NODE_PARAMS_v1, using CUDA's individual-parameter pointer ABI.
#[repr(C)]
#[derive(Debug)]
pub struct KernelNodeParameters {
    pub function: *mut c_void,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub parameters: *mut *mut c_void,
    pub extra: *mut *mut c_void,
}

impl Graph {
    /// Captures raw CUDA operations on `stream` in thread-local capture mode.
    ///
    /// # Safety
    /// The closure must obey CUDA capture restrictions and retain all referenced resources
    /// across every replay. It must not panic or leave joined streams unjoined.
    pub unsafe fn capture(stream: &Stream, submit: impl FnOnce()) -> Self {
        unsafe {
            Self::try_capture(stream, || {
                submit();
                Ok(())
            })
        }
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The resource and lifetime requirements of [`Self::capture`] apply.
    pub unsafe fn try_capture(
        stream: &Stream,
        submit: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        let context = Context::try_current()?;
        ffi::result(
            unsafe { ffi::cuStreamBeginCapture_v2(stream.raw, 1) },
            "cuStreamBeginCapture",
        )?;
        let submitted = submit();
        let mut raw = ptr::null_mut();
        let ended = ffi::result(
            unsafe { ffi::cuStreamEndCapture(stream.raw, &mut raw) },
            "cuStreamEndCapture",
        );
        if submitted.is_err() || ended.is_err() {
            if !raw.is_null() {
                unsafe { ffi::cuGraphDestroy(raw) };
            }
            submitted?;
            ended?;
        }
        Ok(Self {
            raw,
            context,
            ownership: Ownership::Owned,
        })
    }

    /// # Safety
    /// The graph must belong to the current context. Borrowed handles and all referenced
    /// resources must outlive uses; owned handles transfer destruction rights.
    pub unsafe fn from_raw(raw: *mut c_void, ownership: Ownership) -> Self {
        Self {
            raw,
            context: Context::current(),
            ownership,
        }
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn instantiate(&self) -> GraphExec {
        self.try_instantiate()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_instantiate(&self) -> Result<GraphExec> {
        let _context = self.context.try_enter()?;
        let mut raw = ptr::null_mut();
        ffi::result(
            unsafe { ffi::cuGraphInstantiateWithFlags(&mut raw, self.raw, 0) },
            "cuGraphInstantiateWithFlags",
        )?;
        Ok(GraphExec {
            raw,
            context: self.context.clone(),
        })
    }

    pub fn kernel_nodes(&self) -> Vec<*mut c_void> {
        self.try_kernel_nodes()
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_kernel_nodes(&self) -> Result<Vec<*mut c_void>> {
        let _context = self.context.try_enter()?;
        let mut count = 0;
        ffi::result(
            unsafe { ffi::cuGraphGetNodes(self.raw, ptr::null_mut(), &mut count) },
            "cuGraphGetNodes",
        )?;
        let mut nodes = vec![ptr::null_mut(); count];
        ffi::result(
            unsafe { ffi::cuGraphGetNodes(self.raw, nodes.as_mut_ptr(), &mut count) },
            "cuGraphGetNodes",
        )?;
        nodes.truncate(count);
        let mut kernels = Vec::new();
        for node in nodes {
            let mut kind = 0;
            ffi::result(
                unsafe { ffi::cuGraphNodeGetType(node, &mut kind) },
                "cuGraphNodeGetType",
            )?;
            if kind == 0 {
                kernels.push(node);
            }
        }
        Ok(kernels)
    }
}

impl GraphExec {
    pub fn try_upload(&self, stream: &Stream) -> Result<()> {
        let _context = self.context.try_enter()?;
        ffi::result(
            unsafe { ffi::cuGraphUpload(self.raw, stream.raw) },
            "cuGraphUpload",
        )
    }

    pub fn upload(&self, stream: &Stream) {
        self.try_upload(stream)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// # Safety
    /// The graph's resources and its memory-safety obligations must hold for this replay.
    pub unsafe fn launch(&self, stream: &Stream) {
        unsafe { self.try_launch(stream) }.unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The resource and lifetime requirements of [`Self::launch`] apply.
    pub unsafe fn try_launch(&self, stream: &Stream) -> Result<()> {
        let _context = self.context.try_enter()?;
        ffi::result(
            unsafe { ffi::cuGraphLaunch(self.raw, stream.raw) },
            "cuGraphLaunch",
        )?;
        Ok(())
    }

    /// Returns CUDA's update result; zero means success, nonzero leaves the executable intact.
    pub fn update(&mut self, graph: &Graph) -> i32 {
        self.try_update(graph)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_update(&mut self, graph: &Graph) -> Result<i32> {
        let _context = self.context.try_enter()?;
        let mut node = ptr::null_mut();
        let mut result = 0;
        let error = unsafe { ffi::cuGraphExecUpdate(self.raw, graph.raw, &mut node, &mut result) };
        if error != 910 {
            ffi::result(error, "cuGraphExecUpdate")?;
        }
        Ok(result)
    }

    /// # Safety
    /// The node must belong to the instantiating graph. Parameters must satisfy its ABI,
    /// safety obligations, CUDA update restrictions, and resource lifetime requirements.
    pub unsafe fn set_kernel(&mut self, node: *mut c_void, parameters: &KernelNodeParameters) {
        unsafe { self.try_set_kernel(node, parameters) }.unwrap_or_else(|error| panic!("{error}"))
    }

    /// # Safety
    /// The resource and lifetime requirements of [`Self::set_kernel`] apply.
    pub unsafe fn try_set_kernel(
        &mut self,
        node: *mut c_void,
        parameters: &KernelNodeParameters,
    ) -> Result<()> {
        let _context = self.context.try_enter()?;
        ffi::result(
            unsafe { ffi::cuGraphExecKernelNodeSetParams(self.raw, node, parameters) },
            "cuGraphExecKernelNodeSetParams",
        )?;
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let Ok(_context) = self.context.try_enter() else {
            return;
        };
        if self.ownership == Ownership::Owned {
            unsafe { ffi::cuGraphDestroy(self.raw) };
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        let Ok(_context) = self.context.try_enter() else {
            return;
        };
        unsafe { ffi::cuGraphExecDestroy(self.raw) };
    }
}
