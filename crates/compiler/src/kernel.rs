use gpurs_artifact::{Clause, Extent, Parameter};

use crate::{
    layout::{self, PassMode, Shape},
    stable::{Instance, Sess},
    verify::{Certificate, Warning},
};

#[derive(Clone)]
pub struct KernelParameter {
    pub layout: Shape,
    pub metadata: Parameter,
}

pub struct KernelPlan {
    pub instance: Instance,
    pub key: String,
    pub symbol: String,
    pub parameters: Vec<KernelParameter>,
    pub extent: Extent,
    pub requirements: Vec<Clause>,
    pub preconditions: Vec<Clause>,
    /// The checks the certified entry point omits; `None` when it does not exist.
    pub certificate: Option<Certificate>,
    /// Whether the certified entry point records arithmetic failures in the status word.
    pub status: bool,
    /// Whether the instrumented entry point exists.
    pub instrumented: bool,
    /// Whether the kernel reads the linear thread index.
    pub uses_index: bool,
    /// Whether the kernel issues asynchronous copies, which must land before the block exits.
    pub asynchronous: bool,
    /// The kernel's entry in the `*.preconditions` report.
    pub summary: String,
    pub warnings: Vec<Warning>,
}

#[derive(Clone)]
pub struct Signature {
    pub arguments: Vec<(PassMode, Shape)>,
    pub result: (PassMode, Shape),
}

impl Signature {
    pub fn of(sess: Sess<'_>, instance: Instance) -> Self {
        let body = sess.body(instance);
        let shape = |ty| {
            let shape = Shape::of(sess, ty);
            (layout::pass_mode(&shape), shape)
        };
        Signature {
            arguments: body
                .arg_locals()
                .iter()
                .map(|local| shape(local.ty))
                .collect(),
            result: shape(body.ret_local().ty),
        }
    }
}
