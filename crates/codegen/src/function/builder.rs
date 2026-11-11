use std::rc::Rc;

use gpurs_ir::{Type, Value as IRValue};
use gpurs_rustc::{
    Body, CrateDef, FxHashMap, Instance, Operand as MirOperand, Place as MirPlace, RigidTy,
    Session, Shape, Span, Ty,
};

use crate::{
    function::unroll::Loop,
    layout,
    module::{CallPath, ModuleBuilder},
    place::{Operand, Place, Value},
};

/// Lowers the MIR body of one monomorphic instance to an NVVM IR function.
pub struct FunctionBuilder<'m, 'sess> {
    pub module: &'m mut ModuleBuilder<'sess>,
    pub sess: Session<'sess>,
    pub instance: Instance,
    pub body: Rc<Body>,
    pub locals: Vec<Place>,
    /// The size of each local held in its own stack slot, for its lifetime markers.
    pub(super) slots: Vec<Option<u64>>,
    pub fast_math: bool,
    pub span: Span,
    pub ir: gpurs_ir::Function,
    /// The IR block of each MIR block, in the first copy of every unrolled loop.
    pub(super) blocks: Vec<gpurs_ir::Block>,
    /// The loops lowered as one copy of their body per iteration, outermost first.
    pub(super) unrolled: Vec<Loop>,
    /// The copy of each unrolled loop enclosing the current block, outermost first.
    pub(crate) copy: Vec<u64>,
    /// The IR blocks of further copies, and the copies awaiting lowering.
    pub(super) copies: FxHashMap<(usize, Vec<u64>), gpurs_ir::Block>,
    pub(super) pending: Vec<(usize, Vec<u64>)>,
    /// Whether the launch has one block and thread dimension.
    pub linear: bool,
    pub aligned: bool,
    pub block: usize,
    pub path: CallPath,
}

impl<'m, 'sess> FunctionBuilder<'m, 'sess> {
    pub fn new(module: &'m mut ModuleBuilder<'sess>, instance: Instance) -> Self {
        let sess = module.sess;
        let body = sess.body(instance);
        let span = body.span;
        FunctionBuilder {
            sess,
            fast_math: sess.is_fast_math(instance.def.def_id()),
            module,
            instance,
            body,
            locals: Vec::new(),
            slots: Vec::new(),
            span,
            ir: gpurs_ir::Function::new("", Type::Void, Vec::new()),
            blocks: Vec::new(),
            unrolled: Vec::new(),
            copy: Vec::new(),
            copies: FxHashMap::default(),
            pending: Vec::new(),
            linear: false,
            aligned: false,
            block: 0,
            path: Vec::new(),
        }
    }

    pub fn shape_of(&self, ty: Ty) -> Shape {
        self.sess.shape(ty)
    }

    /// The address `offset` bytes into `place`, with the alignment it keeps there.
    pub fn interior(&mut self, place: &Place, offset: u64) -> (IRValue, u64) {
        let pointer = self.offset(place.pointer.clone(), offset);
        (pointer, layout::restrict_align(place.align, offset))
    }

    /// Stores `value`, which has the layout of `destination`, into it.
    pub fn assign(&mut self, destination: &Place, value: Value) {
        let operand = Operand {
            value,
            shape: destination.shape.clone(),
        };
        self.store(destination, &operand);
    }

    /// The element layout of an array, slice, reference, or single-field aggregate.
    pub fn element_shape(&self, place: &Place) -> Shape {
        match gpurs_rustc::rigid(place.shape.ty) {
            RigidTy::Array(element, _)
            | RigidTy::Slice(element)
            | RigidTy::Ref(_, element, _)
            | RigidTy::RawPtr(element, _) => self.shape_of(element),
            _ => self.sess.field_shape(&place.shape, 0),
        }
    }

    /// A stack slot allocated in the entry block.
    pub fn temporary(&mut self, shape: Shape) -> Place {
        let pointer = self
            .ir
            .alloca(shape.size().bytes().max(1) as u64, shape.align());
        Place::new(pointer, shape)
    }

    pub fn error(&self, message: impl Into<String>) {
        self.sess.error(self.span, message).emit();
    }

    pub fn unsupported(&self, what: &str) {
        self.sess
            .error(self.span, format!("{what} is not supported in GPU code"))
            .with_help(
                "use a supported Rust operation or move this operation to the host before \
                 launching the kernel",
            )
            .emit();
    }

    /// Ends a block that verification proved unreachable; an audit build traps there instead.
    pub fn diverge(&mut self) {
        if self.module.audit {
            self.ir.assembly("trap;", "", true, Type::Void, Vec::new());
        }
        self.ir.unreachable();
    }

    pub fn branch(&mut self, target: usize) {
        let target = self.target(target);
        self.ir.branch(target);
    }

    pub fn operand(&mut self, operand: &MirOperand) -> Operand {
        match operand {
            MirOperand::Copy(place) | MirOperand::Move(place) => {
                let place = self.place(place);
                self.load(&place)
            }
            MirOperand::Constant(constant) => self.constant(constant),
            MirOperand::RuntimeChecks(_) => {
                Operand::immediate(IRValue::bool(false), self.shape_of(Ty::bool_ty()))
            }
        }
    }

    /// The type of a MIR place, following its projections.
    pub fn place_type(&self, place: &MirPlace) -> Ty {
        place
            .ty(self.body.locals())
            .expect("places of monomorphic MIR have types")
    }
}
