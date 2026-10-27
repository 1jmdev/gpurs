use std::{env, fs, path::PathBuf};

use gpurs_artifact::{Artifact, Module};
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::{interface::Compiler, util};

use crate::{
    attributes,
    codegen::ModuleBuilder,
    discovery::HostView,
    nvvm, paths,
    stable::{
        self, CrateDef, DefId, GenericArgKind, GenericArgs, Instance, Sess, Session, Ty, TyCtxt,
    },
    transfer, verify,
};

pub struct DeviceView {
    pub crate_name: String,
    pub host: HostView,
}

impl DeviceView {
    fn compile(&self, sess: Sess<'_>) -> Result<(), String> {
        let architecture = env::var("GPURS_ARCHITECTURE").unwrap();
        let directory = PathBuf::from(env::var_os("GPURS_ARTIFACTS").unwrap());
        let definitions: Vec<(DefId, String)> = rustc_public::all_local_items()
            .into_iter()
            .filter_map(|item| {
                attributes::marker(sess, item.def_id(), "device").map(|tail| (item.def_id(), tail))
            })
            .collect();
        let mut module = ModuleBuilder::new(sess);
        let mut proven = ModuleBuilder::new(sess);
        proven.checked = false;
        let mut kernels = Vec::new();
        for launch in &self.host.launches {
            let Some((def_id, _)) = definitions.iter().find(|(def_id, tail)| {
                *tail == launch.tail && paths::module_path(sess, *def_id) == launch.module
            }) else {
                sess.err(format!(
                    "GPU definition for `{}` was not found in the device view",
                    launch.key
                ));
                continue;
            };
            let arguments = launch
                .generics
                .iter()
                .map(|argument| transfer::restore(sess, argument))
                .collect::<Option<Vec<_>>>();
            let Some(arguments) = arguments else {
                sess.err(format!(
                    "cannot restore the concrete types of `{}` in the device view",
                    launch.key
                ));
                continue;
            };
            let definition = rustc_public::all_local_items()
                .into_iter()
                .find(|item| item.def_id() == *def_id)
                .expect("device definition is a local item");
            let function = definition.ty();
            let kind = function.kind();
            let Some((function, signature)) = kind.fn_def() else {
                sess.err(format!("`{}` is not a function", launch.key));
                continue;
            };
            let mut arguments = arguments.into_iter();
            let concrete = GenericArgs(
                signature
                    .0
                    .iter()
                    .map(|argument| match argument {
                        GenericArgKind::Lifetime(region) => {
                            GenericArgKind::Lifetime(region.clone())
                        }
                        GenericArgKind::Const(value) => GenericArgKind::Const(value.clone()),
                        GenericArgKind::Type(_) => {
                            GenericArgKind::Type(arguments.next().unwrap_or_else(|| {
                                sess.err(format!("not enough type arguments for `{}`", launch.key));
                                Ty::bool_ty()
                            }))
                        }
                    })
                    .collect(),
            );
            let instance = match Instance::resolve(function, &concrete) {
                Ok(instance) => instance,
                Err(error) => {
                    sess.err(format!("cannot instantiate `{}`: {error:?}", launch.key));
                    continue;
                }
            };
            if let Some(plan) = verify::kernel(
                sess,
                instance,
                launch.configured,
                launch.key.clone(),
                format!("gpurs_kernel_{}", kernels.len()),
            ) {
                if plan.infallible {
                    proven.kernel(plan.clone());
                }
                kernels.push(module.kernel(plan));
            }
        }
        sess.abort_if_errors();
        let mut artifact = Artifact {
            compiler: format!(
                "gpurs-{} {}",
                env!("CARGO_PKG_VERSION"),
                util::rustc_version_str().unwrap_or("unknown rustc")
            ),
            architecture: architecture.clone(),
            types: self.host.types.clone(),
            modules: Vec::new(),
        };
        if !kernels.is_empty() {
            let ir = module.finish();
            sess.abort_if_errors();
            let toolkit = nvvm::toolkit()?;
            let scratch = directory.join(&self.crate_name);
            fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
            let ptx = nvvm::ptx(&toolkit, &ir, &architecture)?;
            let cubin = nvvm::cubin(&toolkit, &ptx, &architecture, &scratch)?;
            let infallible = if kernels.iter().any(|kernel| kernel.infallible.is_some()) {
                let proven = nvvm::ptx(&toolkit, &proven.finish(), &architecture)?;
                nvvm::cubin(&toolkit, &proven, &architecture, &scratch)?
            } else {
                Vec::new()
            };
            fs::remove_file(scratch.join("kernels.ptx")).map_err(|error| error.to_string())?;
            fs::remove_file(scratch.join("kernels.cubin")).map_err(|error| error.to_string())?;
            artifact.modules.push(Module {
                cubin,
                infallible,
                kernels,
            });
        }
        fs::write(
            directory.join(format!("{}.gpurs", self.crate_name)),
            artifact.encode(),
        )
        .map_err(|error| error.to_string())
    }
}

impl Callbacks for DeviceView {
    fn after_expansion(&mut self, _: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        stable::run(tcx, || {
            let session = Session::new(tcx);
            if let Err(message) = self.compile(session) {
                session.err(message);
            }
            session.abort_if_errors();
        });
        Compilation::Stop
    }
}
