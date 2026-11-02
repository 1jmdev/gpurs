use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs, io, ptr,
};

use libloading::Library;

use crate::{
    Toolkit,
    toolkit::{LIBDEVICE, LIBNVVM},
};

type Handle = *mut c_void;

struct Program {
    handle: Handle,
    destroy: unsafe extern "C" fn(*mut Handle) -> c_int,
}

pub struct Compiler {
    library: Library,
    libdevice: Vec<u8>,
}

impl Toolkit {
    pub fn compiler(&self) -> io::Result<Compiler> {
        // SAFETY: the library remains loaded for the lifetime of every program.
        let library = unsafe { Library::new(self.root.join(LIBNVVM)) }
            .map_err(|error| io::Error::other(format!("cannot load libNVVM: {error}")))?;
        Ok(Compiler {
            library,
            libdevice: fs::read(self.root.join(LIBDEVICE))?,
        })
    }
}

impl Compiler {
    /// Compiles NVVM IR to PTX for `sm_XY`, linking libdevice lazily.
    ///
    /// # Errors
    /// Returns an error when loading CUDA libraries or compiling IR fails.
    pub fn ptx(&self, ir: &str, architecture: &str) -> io::Result<String> {
        // SAFETY: entry points match nvvm.h; buffers and the library outlive the program.
        unsafe {
            macro_rules! entry {
                ($name:literal: $($argument:ty),*) => {
                    *self.library
                        .get::<unsafe extern "C" fn($($argument),*) -> c_int>($name)
                        .map_err(io::Error::other)?
                };
            }
            let create = entry!(b"nvvmCreateProgram": *mut Handle);
            let destroy = entry!(b"nvvmDestroyProgram": *mut Handle);
            let add =
                entry!(b"nvvmAddModuleToProgram": Handle, *const c_char, usize, *const c_char);
            let lazy =
                entry!(b"nvvmLazyAddModuleToProgram": Handle, *const c_char, usize, *const c_char);
            let compile = entry!(b"nvvmCompileProgram": Handle, c_int, *const *const c_char);
            let log_size = entry!(b"nvvmGetProgramLogSize": Handle, *mut usize);
            let log = entry!(b"nvvmGetProgramLog": Handle, *mut c_char);
            let result_size = entry!(b"nvvmGetCompiledResultSize": Handle, *mut usize);
            let result = entry!(b"nvvmGetCompiledResult": Handle, *mut c_char);

            let mut handle = ptr::null_mut();
            check(create(&mut handle))?;
            let program = Program { handle, destroy };
            check(add(handle, ir.as_ptr().cast(), ir.len(), c"gpurs".as_ptr()))?;
            check(lazy(
                handle,
                self.libdevice.as_ptr().cast(),
                self.libdevice.len(),
                c"libdevice".as_ptr(),
            ))?;
            let architecture = CString::new(format!(
                "-arch=compute_{}",
                architecture.trim_start_matches("sm_")
            ))?;
            let options: [&CStr; 6] = [
                &architecture,
                c"-opt=3",
                c"-fma=1",
                c"-prec-div=1",
                c"-prec-sqrt=1",
                c"-ftz=0",
            ];
            let options = options.map(CStr::as_ptr);
            let status = compile(handle, options.len() as c_int, options.as_ptr());
            let read =
                |size: unsafe extern "C" fn(Handle, *mut usize) -> c_int,
                 get: unsafe extern "C" fn(Handle, *mut c_char) -> c_int| {
                    let mut length = 0;
                    check(size(program.handle, &mut length))?;
                    let mut buffer = vec![0u8; length.max(1)];
                    check(get(program.handle, buffer.as_mut_ptr().cast()))?;
                    Ok(CStr::from_bytes_until_nul(&buffer)
                        .map(|text| text.to_string_lossy().into_owned())
                        .unwrap_or_default())
                };
            match status {
                0 => read(result_size, result),
                _ => Err(io::Error::other(format!(
                    "libNVVM rejected the generated IR:\n{}",
                    read(log_size, log)?
                ))),
            }
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: the handle belongs to this program and its library is still loaded.
        unsafe { (self.destroy)(&mut self.handle) };
    }
}

fn check(status: c_int) -> io::Result<()> {
    match status {
        0 => Ok(()),
        _ => Err(io::Error::other(format!(
            "libNVVM failed with status {status}"
        ))),
    }
}
