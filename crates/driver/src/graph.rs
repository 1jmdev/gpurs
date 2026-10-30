use std::{ffi::c_void, ptr};

use crate::{Context, Ownership, Stream, ffi};

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
        let context = Context::current();
        ffi::check(
            unsafe { ffi::cuStreamBeginCapture_v2(stream.raw, 1) },
            "cuStreamBeginCapture",
        );
        submit();
        let mut raw = ptr::null_mut();
        ffi::check(
            unsafe { ffi::cuStreamEndCapture(stream.raw, &mut raw) },
            "cuStreamEndCapture",
        );
        Self {
            raw,
            context,
            ownership: Ownership::Owned,
        }
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
        let _context = self.context.enter();
        let mut raw = ptr::null_mut();
        ffi::check(
            unsafe { ffi::cuGraphInstantiateWithFlags(&mut raw, self.raw, 0) },
            "cuGraphInstantiateWithFlags",
        );
        GraphExec {
            raw,
            context: self.context.clone(),
        }
    }

    pub fn kernel_nodes(&self) -> Vec<*mut c_void> {
        let _context = self.context.enter();
        let mut count = 0;
        ffi::check(
            unsafe { ffi::cuGraphGetNodes(self.raw, ptr::null_mut(), &mut count) },
            "cuGraphGetNodes",
        );
        let mut nodes = vec![ptr::null_mut(); count];
        ffi::check(
            unsafe { ffi::cuGraphGetNodes(self.raw, nodes.as_mut_ptr(), &mut count) },
            "cuGraphGetNodes",
        );
        nodes.truncate(count);
        nodes.retain(|&node| {
            let mut kind = 0;
            ffi::check(
                unsafe { ffi::cuGraphNodeGetType(node, &mut kind) },
                "cuGraphNodeGetType",
            );
            kind == 0
        });
        nodes
    }
}

impl GraphExec {
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    /// # Safety
    /// The graph's resources and its memory-safety obligations must hold for this replay.
    pub unsafe fn launch(&self, stream: &Stream) {
        let _context = self.context.enter();
        ffi::check(
            unsafe { ffi::cuGraphLaunch(self.raw, stream.raw) },
            "cuGraphLaunch",
        );
    }

    /// Returns CUDA's update result; zero means success, nonzero leaves the executable intact.
    pub fn update(&mut self, graph: &Graph) -> i32 {
        let _context = self.context.enter();
        let mut node = ptr::null_mut();
        let mut result = 0;
        let error = unsafe { ffi::cuGraphExecUpdate(self.raw, graph.raw, &mut node, &mut result) };
        if result == 0 {
            ffi::check(error, "cuGraphExecUpdate");
        }
        result
    }

    /// # Safety
    /// The node must belong to the instantiating graph. Parameters must satisfy its ABI,
    /// safety obligations, CUDA update restrictions, and resource lifetime requirements.
    pub unsafe fn set_kernel(&mut self, node: *mut c_void, parameters: &KernelNodeParameters) {
        let _context = self.context.enter();
        ffi::check(
            unsafe { ffi::cuGraphExecKernelNodeSetParams(self.raw, node, parameters) },
            "cuGraphExecKernelNodeSetParams",
        );
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _context = self.context.enter();
        if self.ownership == Ownership::Owned {
            unsafe { ffi::cuGraphDestroy(self.raw) };
        }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        let _context = self.context.enter();
        unsafe { ffi::cuGraphExecDestroy(self.raw) };
    }
}
