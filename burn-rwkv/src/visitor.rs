use std::marker::PhantomData;

use burn_core::module::{Module, ModuleMapper, ModuleVisitor, ParamId};
use burn_tensor::{backend::Backend, BasicOps, Shape, Tensor, TensorKind};

#[macro_export]
macro_rules! module_fn {
    (   visit=$module:expr,
        id=$id:expr,
        state=$state_ty:ty,
        init=$init:expr,
        fn=$fn:expr
    ) => {{
        struct Visitor<'a, B: Backend> {
            state: &'a mut $state_ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleVisitor<B> for Visitor<'a, B> {
            fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
                if $id.eq(id) {
                    let func = $fn;
                    func(tensor, &mut self.state)
                }
            }
        }
        let mut state = $init();
        let mut visitor = Visitor {
            state: &mut state,
            backend: core::marker::PhantomData,
        };
        $module.visit(&mut visitor);
        state
    }};
    (   map=$module:expr,
        id=$id:expr,
        args=$args_ty:ty,
        init=$init:expr,
        fn=$fn:expr
    ) => {{
        struct Mapper<'a, B: Backend> {
            args: &'a mut $args_ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleMapper<B> for Mapper<'a, B> {
            fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
                if $id.eq(id) {
                    let func = $fn;
                    func(self.args, tensor)
                } else {
                    tensor
                }
            }
        }
        let mut args = $init();
        let mut mapper = Mapper {
            args: &mut args,
            backend: core::marker::PhantomData,
        };
        $module.map(&mut mapper)
    }};
}

// struct ParamVisitor<'a, const D: usize, B: Backend, F> {
//     id: Option<&'a ParamId>,
//     func: F,
//     _t: PhantomData<Tensor<B, D>>,
// }
// impl<'a, const D: usize, B: Backend, F> ModuleVisitor<B> for ParamVisitor<'a, D, B, F>
// where
//     F: Fn(&mut Self, &Tensor<B, D>),
// {
//     fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
//         match self.id {
//             Some(target) if target == id => {}
//         }
//     }
// }

// ///
// pub trait View<B: Backend, const D: usize> {
//     ///
//     fn dims(&self, name: &str) -> [usize; D];

//     // ///
//     // fn view<T>(&self, name: &str) -> &T;
// }

// struct ParamIdMapper<F>(F);
// impl<B: Backend, F> ModuleMapper<B> for ParamIdMapper<F>
// where
//     F: FnOnce(&Tensor<B, D>) -> Tensor<B, D>,
// {
//     fn map<const D: usize>(&mut self, id: &ParamId, mut tensor: Tensor<B, D>) -> Tensor<B, D> {
//         self.0(&tensor)
//     }
// }

// impl<B, const D: usize, K> View<B, D> for Tensor<B, D, K>
// where
//     B: Backend,
//     K: TensorKind<B> + BasicOps<B>,
// {
//     fn dims(&self, name: &str) -> [usize; D] {
//         self.dims()
//     }
//     fn view(&self, name: S) -> &Tensor<B, D, K> {
//         Tensor::view(self, name)
//     }
// }
