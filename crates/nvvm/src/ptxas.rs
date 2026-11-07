use std::{
    fs, io,
    path::Path,
    process::{Command, Output},
};

use crate::{
    Toolkit,
    toolkit::{CUOBJDUMP, PTXAS},
};

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
        self.ptxas(ptx, architecture, registers, directory, &[])?;
        fs::read(directory.join("kernels.cubin"))
    }

    /// Assembles PTX like [`Toolkit::assemble`] and returns the `ptxas -v` resource report.
    ///
    /// # Errors
    /// Returns an error on file operations or a failed ptxas invocation.
    pub fn report(
        &self,
        ptx: &str,
        architecture: &str,
        registers: Option<&str>,
        directory: &Path,
    ) -> io::Result<String> {
        let output = self.ptxas(ptx, architecture, registers, directory, &["-v"])?;
        Ok(String::from_utf8_lossy(&output.stderr).into_owned())
    }

    /// Disassembles a cubin file to SASS.
    ///
    /// # Errors
    /// Returns an error when cuobjdump cannot run or fails.
    pub fn disassemble(&self, cubin: &Path) -> io::Result<String> {
        self.cuobjdump("-sass", cubin)
    }

    /// The `cuobjdump -res-usage` resource report of a cubin file.
    ///
    /// # Errors
    /// Returns an error when cuobjdump cannot run or fails.
    pub fn resources(&self, cubin: &Path) -> io::Result<String> {
        self.cuobjdump("-res-usage", cubin)
    }

    fn ptxas(
        &self,
        ptx: &str,
        architecture: &str,
        registers: Option<&str>,
        directory: &Path,
        flags: &[&str],
    ) -> io::Result<Output> {
        let source = directory.join("kernels.ptx");
        fs::write(&source, ptx)?;
        let mut command = Command::new(self.root.join(PTXAS));
        if let Some(registers) = registers {
            command.arg(format!("--maxrregcount={registers}"));
        }
        let result = command
            .args(flags)
            .arg(format!("-arch={architecture}"))
            .arg("-O3")
            .arg(&source)
            .arg("-o")
            .arg(directory.join("kernels.cubin"))
            .output()
            .map_err(|error| io::Error::other(format!("cannot run ptxas: {error}")))?;
        if !result.status.success() {
            return Err(io::Error::other(format!(
                "ptxas failed:\n{}",
                String::from_utf8_lossy(&result.stderr)
            )));
        }
        Ok(result)
    }

    fn cuobjdump(&self, flag: &str, cubin: &Path) -> io::Result<String> {
        let result = Command::new(self.root.join(CUOBJDUMP))
            .arg(flag)
            .arg(cubin)
            .output()
            .map_err(|error| io::Error::other(format!("cannot run cuobjdump: {error}")))?;
        if !result.status.success() {
            return Err(io::Error::other(format!(
                "cuobjdump failed:\n{}",
                String::from_utf8_lossy(&result.stderr)
            )));
        }
        Ok(String::from_utf8_lossy(&result.stdout).into_owned())
    }
}
