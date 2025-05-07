module {
  func.func @main() {
    %val = arith.constant 1263 : i32
    call @print_i32(%val) : (i32) -> ()
    return
  }

  func.func private @print_i32(i32)
}
