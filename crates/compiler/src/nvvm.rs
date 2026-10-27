//! libNVVM (NVVM IR to PTX) and ptxas (PTX to CUBIN).

use std::{
    env,
    ffi::{CStr, c_char, c_int, c_void},
    fs,
    path::{Path, PathBuf},
    process::Command,
    ptr,
};

use libloading::Library;

type Program = *mut c_void;

/// Locates the CUDA toolkit through `CUDA_PATH`, `CUDA_HOME`, or the standard install paths.
pub fn toolkit() -> Result<PathBuf, String> {
    ["CUDA_PATH", "CUDA_HOME"]
        .into_iter()
        .filter_map(env::var_os)
        .map(PathBuf::from)
        .chain(["/opt/cuda", "/usr/local/cuda"].map(PathBuf::from))
        .find(|root| root.join("nvvm/lib64/libnvvm.so").exists())
        .ok_or_else(|| "CUDA toolkit with libNVVM not found; set CUDA_PATH".to_owned())
}

/// Compiles NVVM IR text, linked lazily against libdevice, to PTX for `compute_XY`.
pub fn ptx(toolkit: &Path, ir: &str, architecture: &str) -> Result<String, String> {
    // SAFETY: libNVVM is a C library whose entry points match the declared signatures.
    unsafe {
        let library = Library::new(toolkit.join("nvvm/lib64/libnvvm.so"))
            .map_err(|error| format!("cannot load libNVVM: {error}"))?;
        macro_rules! entry {
            ($name:literal: $($argument:ty),*) => {
                *library
                    .get::<unsafe extern "C" fn($($argument),*) -> c_int>($name)
                    .map_err(|error| error.to_string())?
            };
        }
        let create = entry!(b"nvvmCreateProgram": *mut Program);
        let destroy = entry!(b"nvvmDestroyProgram": *mut Program);
        let add = entry!(b"nvvmAddModuleToProgram": Program, *const c_char, usize, *const c_char);
        let lazy =
            entry!(b"nvvmLazyAddModuleToProgram": Program, *const c_char, usize, *const c_char);
        let compile = entry!(b"nvvmCompileProgram": Program, c_int, *const *const c_char);
        let log_size = entry!(b"nvvmGetProgramLogSize": Program, *mut usize);
        let log = entry!(b"nvvmGetProgramLog": Program, *mut c_char);
        let result_size = entry!(b"nvvmGetCompiledResultSize": Program, *mut usize);
        let result = entry!(b"nvvmGetCompiledResult": Program, *mut c_char);

        let libdevice = fs::read(toolkit.join("nvvm/libdevice/libdevice.10.bc"))
            .map_err(|error| format!("cannot read libdevice: {error}"))?;
        let mut program = ptr::null_mut();
        create(&mut program);
        add(program, ir.as_ptr().cast(), ir.len(), c"gpurs".as_ptr());
        lazy(
            program,
            libdevice.as_ptr().cast(),
            libdevice.len(),
            c"libdevice".as_ptr(),
        );
        let architecture = format!("-arch=compute_{}\0", &architecture[3..]);
        let options: [&CStr; 6] = [
            CStr::from_bytes_with_nul(architecture.as_bytes()).expect("NUL-terminated"),
            c"-opt=3",
            c"-fma=1",
            c"-prec-div=1",
            c"-prec-sqrt=1",
            c"-ftz=0",
        ];
        let options = options.map(CStr::as_ptr);
        let status = compile(program, options.len() as c_int, options.as_ptr());
        let read = |size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
                    get: unsafe extern "C" fn(Program, *mut c_char) -> c_int| {
            let mut length = 0;
            size(program, &mut length);
            let mut buffer = vec![0u8; length.max(1)];
            get(program, buffer.as_mut_ptr().cast());
            CStr::from_bytes_until_nul(&buffer)
                .map(|text| text.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let outcome = match status {
            0 => Ok(read(result_size, result)),
            _ => Err(format!(
                "libNVVM rejected the generated IR:\n{}",
                read(log_size, log)
            )),
        };
        destroy(&mut program);
        outcome
    }
}

/// Assembles PTX into native CUBIN for `sm_XY` with ptxas.
pub fn cubin(
    toolkit: &Path,
    ptx: &str,
    architecture: &str,
    directory: &Path,
) -> Result<Vec<u8>, String> {
    let source = directory.join("kernels.ptx");
    let output = directory.join("kernels.cubin");
    fs::write(&source, ptx).map_err(|error| error.to_string())?;
    let result = Command::new(toolkit.join("bin/ptxas"))
        .arg(format!("-arch={architecture}"))
        .arg("-O3")
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .map_err(|error| format!("cannot run ptxas: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "ptxas failed:\n{}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    fs::read(&output).map_err(|error| error.to_string())
}
