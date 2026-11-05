use std::{
    panic::{self, AssertUnwindSafe},
    process::ExitCode,
};

use gpurs_artifact::Target;
use rustc_driver::{self, Callbacks, Compilation};
use rustc_interface::{
    interface::{Compiler, Config},
    util,
};
use rustc_middle::ty::TyCtxt;

use crate::{Session, session, source::Source};

type Analyze<'a> = (Target, &'a (dyn for<'tcx> Fn(Session<'tcx>) + Sync));

struct CompilationCallbacks<'a> {
    transform: fn(&str, &str) -> String,
    analyze: Option<Analyze<'a>>,
}

impl Callbacks for CompilationCallbacks<'_> {
    fn config(&mut self, config: &mut Config) {
        config.file_loader = Some(Box::new(Source {
            transform: self.transform,
        }));
    }

    fn after_analysis(&mut self, _: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        let Some((target, analyze)) = self.analyze else {
            return Compilation::Continue;
        };
        session::enter(tcx, || {
            let session = Session::new(tcx, target);
            analyze(session);
            session.abort_if_errors();
        });
        Compilation::Stop
    }
}

pub fn run(
    arguments: &[String],
    transform: fn(&str, &str) -> String,
    analyze: Option<Analyze<'_>>,
) -> i32 {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        rustc_driver::catch_with_exit_code(|| {
            #[cfg(gpurs_api_compiler_entrypoint)]
            rustc_driver::compiler_entrypoint(
                arguments,
                &mut CompilationCallbacks { transform, analyze },
            );
            #[cfg(not(gpurs_api_compiler_entrypoint))]
            rustc_driver::run_compiler(arguments, &mut CompilationCallbacks { transform, analyze });
        })
    }));
    match result {
        Ok(ExitCode::SUCCESS) => 0,
        Ok(_) => 1,
        Err(_) => {
            eprintln!("error: gpurs compiler failed while compiling GPU code");
            eprintln!("  = note: this is a compiler error, not an error in your Rust program");
            1
        }
    }
}

pub fn version() -> &'static str {
    util::rustc_version_str().unwrap_or("unknown rustc")
}
