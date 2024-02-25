use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, initial_value: T) -> Self
    where
        T: Clone,
    {
        let size = shape.iter().product();
        let data = vec![initial_value; size];
        Tensor { shape, data }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let index_flat = self.index_to_flat(index);
        &self.data[index_flat]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let index_flat = self.index_to_flat(index);
        &mut self.data[index_flat]
    }
}

impl<T> Tensor<T> {
    fn index_to_flat(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.shape.len());
        let mut flat_index = 0;
        let mut stride = 1;
        for (i, &idx) in index.iter().rev().enumerate() {
            flat_index += idx * stride;
            stride *= self.shape[self.shape.len() - 1 - i];
        }
        flat_index
    }
}

// fn main() {
//     let shape = vec![2, 3, 4];
//     let mut array = Tensor::new(shape.clone(), 0);
//     println!("Array shape: {:?}", shape);
//     println!("Array size: {}", array.size());

//     // Accessing elements
//     array[[0, 1, 2]] = 42;
//     println!("Array[0, 1, 2]: {}", array[[0, 1, 2]]);
// }