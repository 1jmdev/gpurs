use std::fmt::Write;

use gpurs_artifact::Failure;
use rustc_abi::BackendRepr;
use rustc_index::IndexVec;
use rustc_middle::{
    mir::{
        self, AssertKind, BasicBlock, Body, Local, NonDivergingIntrinsic, StatementKind,
        TerminatorKind, traversal,
    },
    ty::{
        EarlyBinder, Instance, Ty, TyCtxt, TypeFoldable, TypingEnv,
        layout::{LayoutOf, TyAndLayout},
    },
};
use rustc_span::Span;

use super::{
    layout::{self, PassMode},
    module::ModuleBuilder,
    place::{Operand, Place, Value},
};
use crate::{attributes, diagnostic};

/// Lowers the MIR body of one monomorphic instance to an NVVM IR function.
pub struct FunctionBuilder<'m, 'tcx> {
    pub module: &'m mut ModuleBuilder<'tcx>,
    pub tcx: TyCtxt<'tcx>,
    pub instance: Instance<'tcx>,
    pub body: &'tcx Body<'tcx>,
    pub locals: IndexVec<Local, Place<'tcx>>,
    pub fast_math: bool,
    pub span: Span,
    pub(super) code: String,
    pub(super) allocas: String,
    pub label: String,
    counter: usize,
}

impl<'m, 'tcx> FunctionBuilder<'m, 'tcx> {
    pub fn new(module: &'m mut ModuleBuilder<'tcx>, instance: Instance<'tcx>) -> Self {
        let tcx = module.tcx;
        let body = tcx.instance_mir(instance.def);
        FunctionBuilder {
            tcx,
            fast_math: attributes::is_fast_math(tcx, instance.def_id()),
            module,
            instance,
            body,
            locals: IndexVec::new(),
            span: body.span,
            code: String::new(),
            allocas: String::new(),
            label: "start".to_owned(),
            counter: 0,
        }
    }

    pub fn monomorphize<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> T {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, value),
        )
    }

    pub fn layout_of(&self, ty: Ty<'tcx>) -> TyAndLayout<'tcx> {
        self.module.cx.layout_of(ty)
    }

    pub fn emit(&mut self, line: impl AsRef<str>) {
        writeln!(self.code, "  {}", line.as_ref()).unwrap();
    }

    /// Emits `%vN = text` and returns `%vN`.
    pub fn instruction(&mut self, text: impl AsRef<str>) -> String {
        self.counter += 1;
        let name = format!("%v{}", self.counter);
        writeln!(self.code, "  {name} = {}", text.as_ref()).unwrap();
        name
    }

    pub fn fresh_label(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    pub fn start_block(&mut self, label: &str) {
        writeln!(self.code, "{label}:").unwrap();
        self.label = label.to_owned();
    }

    /// A stack slot allocated in the entry block.
    pub fn temporary(&mut self, layout: TyAndLayout<'tcx>) -> Place<'tcx> {
        self.counter += 1;
        let name = format!("%s{}", self.counter);
        let size = layout.size.bytes().max(1);
        let align = layout.align.abi.bytes();
        writeln!(
            self.allocas,
            "  {name}.slot = alloca [{size} x i8], align {align}"
        )
        .unwrap();
        writeln!(
            self.allocas,
            "  {name} = bitcast [{size} x i8]* {name}.slot to i8*"
        )
        .unwrap();
        Place::new(name, layout)
    }

    pub fn error(&self, message: impl Into<String>) {
        self.tcx.dcx().span_err(self.span, message.into());
    }

    pub fn unsupported(&self, what: &str) {
        diagnostic::error(
            self.tcx,
            self.span,
            format!("{what} is not supported in GPU code"),
            "use a supported Rust operation or move this operation to the host before launching the kernel",
        );
    }

    /// Records a device failure and terminates the thread.
    pub fn fail(&mut self, failure: Failure) {
        self.emit(format!("call void @gpurs.fail(i32 {})", failure as u32));
        self.emit("unreachable");
    }

    pub fn lower(mut self, name: &str) -> String {
        let signature = self.module.signature(self.instance);
        for (local, declaration) in self.body.local_decls.iter_enumerated() {
            let layout = self.layout_of(self.monomorphize(declaration.ty));
            let place = if layout.is_zst() || layout.is_unsized() {
                Place::new(
                    format!("inttoptr (i64 {} to i8*)", layout.align.abi.bytes()),
                    layout,
                )
            } else if local == mir::RETURN_PLACE && matches!(signature.result.0, PassMode::Indirect)
            {
                Place::new("%result".to_owned(), layout)
            } else if let Some(index) = argument_index(self.body, local)
                && let PassMode::Indirect = signature.arguments[index].0
            {
                Place::new(format!("%a{index}"), layout)
            } else {
                self.temporary(layout)
            };
            self.locals.push(place);
        }
        for (index, local) in self.body.args_iter().enumerate() {
            let place = self.locals[local].clone();
            let value = match signature.arguments[index].0 {
                PassMode::Direct(_) => Value::Scalar(format!("%a{index}")),
                PassMode::Pair(..) => Value::Pair(format!("%a{index}.0"), format!("%a{index}.1")),
                PassMode::Ignore | PassMode::Indirect => continue,
            };
            self.store(
                &place,
                &Operand {
                    value,
                    layout: place.layout,
                },
            );
        }
        self.emit("br label %bb0");

        let reachable = traversal::mono_reachable_as_bitset(self.body, self.tcx, self.instance);
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            if data.is_cleanup {
                continue;
            }
            self.start_block(&format!("bb{}", block.as_usize()));
            if !reachable.contains(block) {
                self.emit("unreachable");
                continue;
            }
            for statement in &data.statements {
                self.span = statement.source_info.span;
                self.statement(&statement.kind);
            }
            self.span = data.terminator().source_info.span;
            self.terminator(&data.terminator().kind, &signature.result);
        }

        let mut function = String::new();
        writeln!(function, "; {}", self.instance).unwrap();
        writeln!(
            function,
            "define internal {} {name}({}) {{\nstart:",
            signature.return_type(),
            signature.parameters()
        )
        .unwrap();
        function.push_str(&self.allocas);
        function.push_str(&self.code);
        function.push_str("}\n\n");
        function
    }

    fn statement(&mut self, statement: &StatementKind<'tcx>) {
        match statement {
            StatementKind::Assign(assign) => {
                let (destination, rvalue) = &**assign;
                let place = self.place(*destination);
                self.rvalue(&place, rvalue);
            }
            StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => {
                let place = self.place(**place);
                self.set_discriminant(&place, *variant_index);
            }
            StatementKind::Intrinsic(intrinsic) => match &**intrinsic {
                NonDivergingIntrinsic::Assume(_) => {}
                NonDivergingIntrinsic::CopyNonOverlapping(copy) => {
                    let source = self.operand(&copy.src);
                    let destination = self.operand(&copy.dst);
                    let count = self.operand(&copy.count);
                    let pointee = destination.layout.ty.builtin_deref(true).expect("pointer");
                    let size = self.layout_of(pointee).size.bytes();
                    let bytes = self.instruction(format!("mul i64 {}, {size}", count.scalar()));
                    self.emit(format!(
                        "call void @llvm.memcpy.p0i8.p0i8.i64(i8* {}, i8* {}, i64 {bytes}, i1 false)",
                        destination.scalar(),
                        source.scalar()
                    ));
                }
            },
            _ => {}
        }
    }

    fn terminator(
        &mut self,
        terminator: &TerminatorKind<'tcx>,
        result: &(PassMode, TyAndLayout<'tcx>),
    ) {
        match terminator {
            TerminatorKind::Goto { target, .. } => self.branch(*target),
            TerminatorKind::FalseEdge { real_target, .. }
            | TerminatorKind::FalseUnwind { real_target, .. } => self.branch(*real_target),
            TerminatorKind::SwitchInt { discr, targets } => {
                let discriminant = self.operand(discr);
                let BackendRepr::Scalar(scalar) = discriminant.layout.backend_repr else {
                    unreachable!("switch on a non-scalar `{}`", discriminant.layout.ty)
                };
                let ty = layout::immediate_type(scalar);
                let cases: Vec<String> = targets
                    .iter()
                    .map(|(value, target)| {
                        let value = integer_text(value, ty, layout::bits(scalar));
                        format!("{ty} {value}, label %bb{}", target.as_usize())
                    })
                    .collect();
                self.emit(format!(
                    "switch {ty} {}, label %bb{} [ {} ]",
                    discriminant.scalar(),
                    targets.otherwise().as_usize(),
                    cases.join(" ")
                ));
            }
            TerminatorKind::Return => {
                let place = self.locals[mir::RETURN_PLACE].clone();
                match result.0 {
                    PassMode::Ignore | PassMode::Indirect => self.emit("ret void"),
                    PassMode::Direct(scalar) => {
                        let value = self.load(&place);
                        self.emit(format!(
                            "ret {} {}",
                            layout::immediate_type(scalar),
                            value.scalar()
                        ));
                    }
                    PassMode::Pair(a, b) => {
                        let value = self.load(&place);
                        let (first, second) = value.pair();
                        let ty = format!(
                            "{{ {}, {} }}",
                            layout::immediate_type(a),
                            layout::immediate_type(b)
                        );
                        let partial = self.instruction(format!(
                            "insertvalue {ty} undef, {} {first}, 0",
                            layout::immediate_type(a)
                        ));
                        let whole = self.instruction(format!(
                            "insertvalue {ty} {partial}, {} {second}, 1",
                            layout::immediate_type(b)
                        ));
                        self.emit(format!("ret {ty} {whole}"));
                    }
                }
            }
            TerminatorKind::Unreachable
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_) => self.emit("unreachable"),
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                ..
            } => {
                if !self.tcx.sess.overflow_checks() && msg.is_optional_overflow_check() {
                    return self.branch(*target);
                }
                let condition = self.operand(cond);
                let failure = self.fresh_label("assert");
                let (on_true, on_false) = match expected {
                    true => (format!("bb{}", target.as_usize()), failure.clone()),
                    false => (failure.clone(), format!("bb{}", target.as_usize())),
                };
                self.emit(format!(
                    "br i1 {}, label %{on_true}, label %{on_false}",
                    condition.scalar()
                ));
                self.start_block(&failure);
                self.fail(match **msg {
                    AssertKind::BoundsCheck { .. } => Failure::BoundsCheck,
                    AssertKind::Overflow(..) | AssertKind::OverflowNeg(_) => Failure::Overflow,
                    AssertKind::DivisionByZero(_) | AssertKind::RemainderByZero(_) => {
                        Failure::DivisionByZero
                    }
                    _ => Failure::Panic,
                });
            }
            TerminatorKind::Drop { place, target, .. } => {
                let ty = self.monomorphize(place.ty(self.body, self.tcx).ty);
                if ty.needs_drop(self.tcx, TypingEnv::fully_monomorphized()) {
                    let place = self.place(*place);
                    let argument = Operand {
                        value: self.address(&place),
                        layout: self.layout_of(Ty::new_mut_ptr(self.tcx, ty)),
                    };
                    let destination = self.temporary(self.layout_of(self.tcx.types.unit));
                    self.invoke(
                        Instance::resolve_drop_glue(self.tcx, ty),
                        &[argument],
                        &destination,
                    );
                }
                self.branch(*target);
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                ..
            } => {
                self.call(func, args, *destination, *target);
            }
            TerminatorKind::TailCall { func, args, .. } => {
                self.call(func, args, mir::RETURN_PLACE.into(), None);
                self.unsupported("explicit tail calls");
            }
            TerminatorKind::InlineAsm { .. } => {
                self.unsupported("inline assembly");
                self.emit("unreachable");
            }
            TerminatorKind::Yield { .. } | TerminatorKind::CoroutineDrop => {
                self.unsupported("coroutines");
                self.emit("unreachable");
            }
        }
    }

    pub fn branch(&mut self, target: BasicBlock) {
        self.emit(format!("br label %bb{}", target.as_usize()));
    }

    pub fn operand(&mut self, operand: &mir::Operand<'tcx>) -> Operand<'tcx> {
        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => {
                let place = self.place(*place);
                self.load(&place)
            }
            mir::Operand::Constant(constant) => self.constant(constant),
            mir::Operand::RuntimeChecks(checks) => Operand {
                value: Value::Scalar(checks.value(self.tcx.sess).to_string()),
                layout: self.layout_of(self.tcx.types.bool),
            },
        }
    }
}

fn argument_index(body: &Body<'_>, local: Local) -> Option<usize> {
    (1..=body.arg_count)
        .contains(&local.as_usize())
        .then(|| local.as_usize() - 1)
}

/// Integer literal of `bits` width, printed as the signed value LLVM expects.
pub fn integer_text(value: u128, ty: &str, bits: u64) -> String {
    if ty == "i1" {
        return (value & 1 == 1).to_string();
    }
    let shift = 128 - bits as u32;
    (((value << shift) as i128) >> shift).to_string()
}
