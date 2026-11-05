use std::{fs, io, path::Path, process::Command};

use crate::{Toolkit, toolkit::PTXAS};

impl Toolkit {
    /// Assembles PTX for `sm_XY`, writing intermediate files into `directory`.
    ///
    /// # Errors
    /// Returns an error on file operations or a failed ptxas invocation.
    pub fn assemble(
        &self,
        ptx: &str,
        architecture: &str,
        registers: Option<&str>,
        directory: &Path,
    ) -> io::Result<Vec<u8>> {
        let source = directory.join("kernels.ptx");
        let output = directory.join("kernels.cubin");
        fs::write(&source, ptx)?;
        let mut command = Command::new(self.root.join(PTXAS));
        if let Some(registers) = registers {
            command.arg(format!("--maxrregcount={registers}"));
        }
        let result = command
            .arg(format!("-arch={architecture}"))
            .arg("-O3")
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .output()
            .map_err(|error| io::Error::other(format!("cannot run ptxas: {error}")))?;
        if !result.status.success() {
            return Err(io::Error::other(format!(
                "ptxas failed:\n{}",
                String::from_utf8_lossy(&result.stderr)
            )));
        }
        fs::read(&output)
    }
}
