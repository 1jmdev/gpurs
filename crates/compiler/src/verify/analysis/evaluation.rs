use super::{
    Analysis, Origin, Pointer, Region, State, Value, integer_kind,
    place::Location,
    range,
    state::{get, set},
};
use crate::{
    policy, stable,
    stable::*,
    verify::symbolic::{Atom, Polynomial},
};

impl Analysis<'_, '_> {
    pub fn statement(&mut self, statement: &StatementKind, state: &mut State) {
        match statement {
            StatementKind::Assign(place, rvalue) => {
                let mut value = self.rvalue(rvalue, state);
                let ty = self.place_type(place);
                self.opaque(
                    &mut value,
                    ty,
                    Vec::new(),
                    "a value the analysis cannot follow",
                );
                self.assign(place, value, state);
            }
            StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => {
                let ty = self.place_type(place);
                let variant = discriminant(ty, *variant_index);
                if let Location::Slot(slot, path) = self.locate(place, state) {
                    let mut value = get(&state.slots[slot], &path);
                    value.variant = variant;
                    value.guards.clear();
                    set(&mut state.slots[slot], &path, value);
                }
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                let slot = self.frame().base + local;
                state.slots[slot] = Value::default();
            }
            StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(_))
                if self.collect && self.quiet == 0 =>
            {
                self.sess
                    .error(self.span, "bulk pointer copies require a GPU access proof");
            }
            _ => {}
        }
    }

    /// Gives an unknown unsigned integer an opaque variable spanning its type.
    pub fn opaque(&mut self, value: &mut Value, ty: Ty, path: Vec<usize>, description: &str) {
        if value.integer.is_some() {
            return;
        }
        if let Some(kind @ (false, _)) = integer_kind(ty) {
            let (_, maximum) = range(kind);
            value.integer = Some(self.variable(
                path,
                Polynomial::default(),
                Some(Polynomial::constant(maximum)),
                crate::verify::engine::opaque(),
                description,
            ));
        }
    }

    pub fn rvalue(&mut self, rvalue: &Rvalue, state: &State) -> Value {
        match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand, state),
            Rvalue::CopyForDeref(place) => self.read(place, state),
            Rvalue::Ref(_, _, place)
            | Rvalue::AddressOf(_, place)
            | Rvalue::Reborrow(_, _, place) => self.address(place, state),
            Rvalue::Cast(kind, operand, target) => self.cast(kind, operand, *target, state),
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => {
                let value = self.operand(operand, state);
                if value.fields.len() == 2 && value.pointer.is_some() {
                    return value.fields[1].clone();
                }
                match value.pointer {
                    Some(Pointer::Memory(Origin {
                        region: Region::Parameter(parameter),
                        index: Some(index),
                        inner: false,
                    })) if index.value() == Some(0) => {
                        Value::integer(Polynomial::atom(Atom::Length(parameter)), 0)
                    }
                    _ => Value::varying(value.variance),
                }
            }
            Rvalue::UnaryOp(UnOp::Not, operand) => {
                let value = self.operand(operand, state);
                if self.operand_type(operand) != Some(Ty::bool_ty()) {
                    return Value::varying(value.variance);
                }
                Value {
                    integer: value
                        .integer
                        .as_ref()
                        .and_then(Polynomial::value)
                        .map(|value| Polynomial::constant(1 - value)),
                    guards: value
                        .guards
                        .iter()
                        .map(|(key, facts)| (1 - key, facts.clone()))
                        .collect(),
                    domain: Some(2),
                    variance: value.variance,
                    ..Value::default()
                }
            }
            Rvalue::UnaryOp(_, operand) => Value::varying(self.operand(operand, state).variance),
            Rvalue::BinaryOp(operation, left, right) => {
                self.binary(*operation, left, right, false, state)
            }
            Rvalue::CheckedBinaryOp(operation, left, right) => {
                self.binary(*operation, left, right, true, state)
            }
            Rvalue::Aggregate(kind, operands) => {
                let fields: Vec<_> = operands
                    .iter()
                    .map(|operand| self.operand(operand, state))
                    .collect();
                let variance = fields.iter().map(|field| field.variance).max().unwrap_or(0);
                match kind {
                    AggregateKind::RawPtr(..) => Value {
                        pointer: fields[0].pointer.clone(),
                        fields,
                        variance,
                        ..Value::default()
                    },
                    AggregateKind::Adt(definition, variant, ..) => Value {
                        variant: (definition.kind() == AdtKind::Enum)
                            .then(|| definition.discriminant_for_variant(*variant).val as i128),
                        domain: (definition.kind() == AdtKind::Enum)
                            .then(|| definition.num_variants()),
                        fields,
                        variance,
                        ..Value::default()
                    },
                    _ => Value {
                        fields,
                        variance,
                        ..Value::default()
                    },
                }
            }
            Rvalue::Repeat(operand, _) => Value::varying(self.operand(operand, state).variance),
            Rvalue::Discriminant(place) => {
                let value = self.read(place, state);
                let domain = match stable::rigid(self.place_type(place)) {
                    RigidTy::Adt(definition, _) => Some(definition.num_variants()),
                    _ => None,
                };
                Value {
                    integer: value.variant.map(Polynomial::constant),
                    guards: value.guards,
                    domain,
                    variance: value.variance,
                    ..Value::default()
                }
            }
            _ => Value::varying(2),
        }
    }

    pub fn operand(&mut self, operand: &Operand, state: &State) -> Value {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.read(place, state),
            Operand::Constant(constant) => {
                let ty = constant.const_.ty();
                let allocation = match constant.const_.kind() {
                    ConstantKind::Allocated(allocation) => allocation.clone(),
                    ConstantKind::Ty(value) => match value.kind() {
                        TyConstKind::Value(_, allocation) => allocation.clone(),
                        _ => return Value::default(),
                    },
                    _ => return Value::default(),
                };
                if let Some(kind) = integer_kind(ty) {
                    let value = if kind.0 {
                        allocation.read_int().ok()
                    } else {
                        allocation.read_uint().ok().map(|value| value as i128)
                    };
                    return Value {
                        integer: value.map(Polynomial::constant),
                        ..Value::default()
                    };
                }
                if ty == Ty::bool_ty() {
                    return Value {
                        integer: allocation
                            .read_bool()
                            .ok()
                            .map(|value| Polynomial::constant(value as i128)),
                        domain: Some(2),
                        ..Value::default()
                    };
                }
                self.aggregate_constant(ty, &allocation)
            }
            Operand::RuntimeChecks(checks) => Value {
                integer: Some(Polynomial::constant(
                    policy::runtime_check(self.sess, checks) as i128,
                )),
                domain: Some(2),
                ..Value::default()
            },
        }
    }

    /// The integer fields of a struct or tuple constant, such as a `Range`.
    fn aggregate_constant(&mut self, ty: Ty, allocation: &Allocation) -> Value {
        if !matches!(
            ty.kind(),
            TyKind::RigidTy(RigidTy::Adt(definition, _)) if definition.kind() == AdtKind::Struct
        ) && !matches!(ty.kind(), TyKind::RigidTy(RigidTy::Tuple(_)))
        {
            return Value::default();
        }
        let shape = crate::codegen::Shape::of(self.sess, ty);
        let bytes = stable::allocation_bytes(allocation);
        let fields = self
            .sess
            .field_types(ty)
            .into_iter()
            .enumerate()
            .map(|(field, element)| {
                let Some(kind @ (false, _)) = integer_kind(element) else {
                    return Value::default();
                };
                let start = shape.offset(field).bytes();
                let size = (kind.1 / 8) as usize;
                let Some(bytes) = bytes.get(start..start + size) else {
                    return Value::default();
                };
                let mut integer = [0u8; 16];
                integer[..size].copy_from_slice(bytes);
                Value {
                    integer: i128::try_from(u128::from_le_bytes(integer))
                        .ok()
                        .map(Polynomial::constant),
                    ..Value::default()
                }
            })
            .collect();
        Value {
            fields,
            ..Value::default()
        }
    }
}

/// The discriminant value of `variant` of the enum `ty`.
pub fn discriminant(ty: Ty, variant: VariantIdx) -> Option<i128> {
    match stable::rigid(ty) {
        RigidTy::Adt(definition, _) if definition.kind() == AdtKind::Enum => {
            Some(definition.discriminant_for_variant(variant).val as i128)
        }
        _ => None,
    }
}
