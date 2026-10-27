use std::{env, fs, path::PathBuf};

use gpurs_artifact::Artifact;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::{interface::Compiler, util};

use crate::{
    attributes,
    cache::Cache,
    discovery::HostView,
    paths,
    stable::{self, CrateDef, GenericArgKind, GenericArgs, Instance, Sess, Session, Ty, TyCtxt},
};

pub struct DeviceView;

impl DeviceView {
    fn compile(&self, sess: Sess<'_>) -> Result<(), String> {
        let mut host = HostView::default();
        host.analyze(sess);
        let architecture = env::var("GPURS_ARCHITECTURE").unwrap();
        let directory = PathBuf::from(env::var_os("GPURS_ARTIFACTS").unwrap());
        let output = PathBuf::from(env::var_os("GPURS_TARGET_ARTIFACT").unwrap());
        let definitions: Vec<_> = rustc_public::all_local_items()
            .into_iter()
            .filter_map(|item| {
                attributes::marker(sess, item.def_id(), "device").map(|tail| (item, tail))
            })
            .collect();
        let mut cache = Cache::new(&directory)?;
        let mut modules = Vec::new();
        let mut report = String::new();
        for launch in &host.launches {
            let Some((definition, _)) = definitions.iter().find(|(definition, tail)| {
                *tail == launch.tail
                    && paths::module_path(sess, definition.def_id())
                        .strip_suffix("::__gpurs_device")
                        == Some(launch.module.as_str())
            }) else {
                sess.err(format!(
                    "GPU definition for `{}` was not found in the device view",
                    launch.key
                ));
                continue;
            };
            let function = definition.ty();
            let kind = function.kind();
            let Some((function, signature)) = kind.fn_def() else {
                sess.err(format!("`{}` is not a function", launch.key));
                continue;
            };
            let mut arguments = launch.generics.iter().copied();
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
            let (module, summary) = cache.kernel(
                sess,
                instance,
                launch.configured,
                &launch.key,
                &architecture,
            )?;
            report.push_str(&summary);
            modules.push(module);
        }
        sess.abort_if_errors();
        let artifact = Artifact {
            compiler: format!(
                "gpurs-{} {}",
                env!("CARGO_PKG_VERSION"),
                util::rustc_version_str().unwrap_or("unknown rustc")
            ),
            architecture,
            types: host.types,
            modules,
        };
        fs::write(output.with_extension("preconditions"), report)
            .map_err(|error| error.to_string())?;
        let bytes = artifact.encode();
        let mut embedded = (bytes.len() as u64).to_le_bytes().to_vec();
        embedded.extend_from_slice(&bytes);
        fs::write(output, embedded).map_err(|error| error.to_string())
    }
}

impl Callbacks for DeviceView {
    fn config(&mut self, config: &mut rustc_interface::interface::Config) {
        config.file_loader = Some(Box::new(crate::source::Source));
    }
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
