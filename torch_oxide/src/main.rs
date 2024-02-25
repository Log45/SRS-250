mod utils;

use utils::tensor::Tensor;

fn main() {
    let shape = vec![2, 3, 4];
    let mut array = Tensor::new(shape.clone(), 0);
    println!("Array shape: {:?}", shape);
    println!("Array size: {}", array.size());

    // Accessing elements
    array[&[0, 1, 2]] = 42;
    println!("Array[0, 1, 2]: {}", array[&[0, 1, 2]]);

    let shape = vec![4, 3];
    let mut array = Tensor::new(shape.clone(), 0);

    // Populating the array
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            array[&[i, j]] = i + j;
        }
    }

    // Printing the array
    println!("Array:");
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            print!("{} ", array[&[i, j]]);
        }
        println!();
    }

    let shape = vec![3, 3, 3];
    let mut array = Tensor::new(shape.clone(), 0);

    // Populating the array
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                array[&[i, j, k]] = i + j + k;
            }
        }
    }

    // Printing the array
    println!("Array:");
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                print!("{} ", array[&[i, j, k]]);
            }
            println!();
        }
        println!();
    }

    let shape = vec![2, 3, 4, 5, 6];
    let mut array = Tensor::new(shape.clone(), 0);

    // Populating the array
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                for l in 0..shape[3] {
                    for m in 0..shape[4] {
                        array[&[i, j, k, l, m]] = i + j + k + l + m;
                    }
                }
            }
        }
    }

    // Printing the array
    println!("Array:");
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                for l in 0..shape[3] {
                    for m in 0..shape[4] {
                        print!("{} ", array[&[i, j, k, l, m]]);
                    }
                    println!();
                }
                println!();
            }
            println!();
        }
        println!();
    }
}
