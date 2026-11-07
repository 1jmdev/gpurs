use gpurs_artifact::KernelEffects;
use gpurs_rustc::{CrateDef, attributes};
use gpurs_verifier::KernelPlan;

use crate::function::FunctionBuilder;

/// Marks the entry as a kernel with its declared thread count and blocks per SM.
pub fn apply(function: &mut FunctionBuilder<'_, '_>, plan: &KernelPlan) {
    function.ir.annotations.push(("kernel", 1));
    if let Some((threads, blocks)) =
        attributes::launch_bounds(function.sess, plan.instance.def.def_id())
    {
        let exchange = plan.effects.contains(KernelEffects::REGISTER_EXCHANGE);
        function.ir.annotations.push(("reqntidx", threads));
        function
            .ir
            .annotations
            .extend((blocks > 0 || exchange).then_some(("minctasm", blocks.max(1))));
    }
}
