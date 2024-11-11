# MJOLNIR README

This serves as documentation for mjolnir devs (me and Edu rn)

## DEI to MLIR
| Instruction Type | Opcode | Opcode name | IR code | Description |
| :---: | :---: | :--- | :--- | :--- |
| Instruction10t | 40  | OP\_GOTO | cf::BranchOp | Jump to a provided offset (positive or negative) |
| Instruction10x | 14  | OP\_RETURN\_VOID | func::ReturnOp | Return operation without any parameter |
|     | 0   | OP\_NOP | Shuriken::MjolnIR::Nop | No-Operation |
| Instruction11n | 18  | OP\_CONST\_4 | Shuriken::MjolnIR::LoadValue | Operation to load the value 4 to a register |
| Instruction11x | 15  | OP\_RETURN | func::ReturnOp | Return operation with register as parameter, it contains 32-bit values |
|     | 16  | OP\_RETURN\_WIDE | func::ReturnOp | Return operation with register as parameter, it contains 64-bit values |
|     | 17  | OP\_RETURN\_OBJECT | func::ReturnOp | Return operation with register as parameter, it contains an object |
|     | 10  | OP\_MOVE\_RESULT | Generate result in Shuriken::MjolnIR::InvokeOp | Move result from an invoke to a register, it contains 32-bit values |
|     | 11  | OP\_MOVE\_RESULT\_WIDE | Generate result in Shuriken::MjolnIR::InvokeOp | Move result from an invoke to a register, it contains 64-bit values |
|     | 12  | OP\_MOVE\_RESULT\_OBJECT | Generate result in Shuriken::MjolnIR::InvokeOp | Move result from an invoke to a register, it contains an object |
| Instruction12x | 1   | OP\_MOVE | Shuriken::MjolnIR::MoveOp | Move a 32-bit value from one register to another |
|     | 4   | OP\_MOVE\_WIDE | Shuriken::MjolnIR::MoveOp | Move a 64-bit value from one register to another |
|     | 7   | OP\_MOVE\_OBJECT | Shuriken::MjolnIR::MoveOp | Move an object from one registe rto another |
|     | 176 | OP\_ADD\_INT\_2ADDR | arith::AddIOp | Add operation |
|     | 187 | OP\_ADD\_LONG\_2ADDR | arith::AddIOp | Add operation |
|     | 198 | OP\_ADD\_FLOAT\_2ADDR | arith::AddIOp | Add operation |
|     | 203 | OP\_ADD\_DOUBLE\_2ADDR | arith::AddIOp | Add operation |
|     | 177 | OP\_SUB\_INT\_2ADDR | arith::SubIOp | Sub operation |
|     | 188 | OP\_SUB\_LONG\_2ADDR | arith::SubIOp | Sub operation |
|     | 199 | OP\_SUB\_FLOAT\_2ADDR | arith::SubIOp | Sub operation |
|     | 204 | OP\_SUB\_DOUBLE\_2ADDR | arith::SubIOp | Sub operation |
|     | 178 | OP\_MUL\_INT\_2ADDR | arith::MulIOp | Mul operation |
|     | 189 | OP\_MUL\_LONG\_2ADDR | arith::MulIOp | Mul operation |
|     | 200 | OP\_MUL\_FLOAT\_2ADDR | arith::MulIOp | Mul operation |
|     | 205 | OP\_MUL\_DOUBLE\_2ADDR | arith::MulIOp | Mul operation |
|     | 179 | OP\_DIV\_INT\_2ADDR | arith::DivSIOp | Div operation |
|     | 190 | OP\_DIV\_LONG\_2ADDR | arith::DivSIOp | Div operation |
|     | 201 | OP\_DIV\_FLOAT\_2ADDR | arith::DivFOp | Div operation |
|     | 206 | OP\_DIV\_DOUBLE\_2ADDR | arith::DivFOp | Div operation |
|     | 180 | OP\_REM\_INT\_2ADDR | arith::RemSIOp | Rem operation |
|     | 191 | OP\_REM\_LONG\_2ADDR | arith::RemSIOp | Rem operation |
|     | 202 | OP\_REM\_FLOAT\_2ADDR | arith::RemFOp | Rem operation |
|     | 207 | OP\_REM\_DOUBLE\_2ADDR | arith::RemFOp | Rem operation |
|     | 181 | OP\_AND\_INT\_2ADDR | arith::AndIOp | And operation |
|     | 192 | OP\_AND\_LONG\_2ADDR | arith::AndIOp | And operation |
|     | 182 | OP\_OR\_INT\_2ADDR | arith::OrIOp | Or operation |
|     | 193 | OP\_OR\_LONG\_2ADDR | arith::OrIOp | Or operation |
|     | 183 | OP\_XOR\_INT\_2ADDR | arith::XOrIOp | Xor operation |
|     | 194 | OP\_XOR\_LONG\_2ADDR | arith::XOrIOp | Xor operation |
|     | 184 | OP\_SHL\_INT\_2ADDR | arith::ShLIOp | Shift-Left operation |
|     | 195 | OP\_SHL\_LONG\_2ADDR | arith::ShLIOp | Shift-Left operation |
|     | 185 | OP\_SHR\_INT\_2ADDR | arith::ShRSIOp | Shift-Right operation |
|     | 196 | OP\_SHR\_LONG\_2ADDR | arith::ShRSIOp | Shift-Right operation |
|     | 186 | OP\_USHR\_INT\_2ADDR | arith::ShRUIOp | Unsigned Shift-Right operation |
|     | 197 | OP\_USHR\_LONG\_2ADDR | arith::ShRUIOp | Unsigned Shift-Right operation |
|     | 123 | OP\_NEG\_INT | Shuriken::MjolnIR::Neg | Neg operation |
|     | 125 | OP\_NEG\_LONG | Shuriken::MjolnIR::Neg | Neg operation |
|     | 127 | OP\_NEG\_FLOAT | Shuriken::MjolnIR::Neg | Neg operation |
|     | 128 | OP\_NEG\_DOUBLE | Shuriken::MjolnIR::Neg | Neg operation |
|     | 124 | OP\_NOT\_INT | Shuriken::MjolnIR::Not | Not operation |
|     | 126 | OP\_NOT\_LONG | Shuriken::MjolnIR::Not | Not operation |
|     | 129 | OP\_INT\_TO\_LONG | MjolnIR::CastOp | Cast from int to long |
|     | 136 | OP\_FLOAT\_TO\_LONG | MjolnIR::CastOp | Cast from float to long |
|     | 139 | OP\_DOUBLE\_TO\_LONG | MjolnIR::CastOp | Cast from double to long |
|     | 130 | OP\_INT\_TO\_FLOAT | MjolnIR::CastOp | Cast from int to float |
|     | 133 | OP\_LONG\_TO\_FLOAT | MjolnIR::CastOp | Cast from long to float |
|     | 140 | OP\_DOUBLE\_TO\_FLOAT | MjolnIR::CastOp | Cast from double to float |
|     | 131 | OP\_INT\_TO\_DOUBLE | MjolnIR::CastOp | Cast from int to double |
|     | 134 | OP\_LONG\_TO\_DOUBLE | MjolnIR::CastOp | Cast from long to double |
|     | 137 | OP\_FLOAT\_TO\_DOUBLE | MjolnIR::CastOp | Cast from float to double |
|     | 132 | OP\_LONG\_TO\_INT | MjolnIR::CastOp | Cast from long to int |
|     | 135 | OP\_FLOAT\_TO\_INT | MjolnIR::CastOp | Cast from float to int |
|     | 138 | OP\_DOUBLE\_TO\_INT | MjolnIR::CastOp | Cast from double to int |
|     | 141 | OP\_INT\_TO\_BYTE | MjolnIR::CastOp | Cast from int to byte |
|     | 142 | OP\_INT\_TO\_CHAR | MjolnIR::CastOp | Cast from int to char |
|     | 143 | OP\_INT\_TO\_SHORT | MjolnIR::CastOp | Cast from int to short |
| Instruction20t | 41  | OP\_GOTO\_16 | cf::BranchOp | Unconditional jump to a provided offset (negative or positive) |
| Instruction21c | 34  | OP\_NEW\_INSTANCE | Shuriken::MjolnIR::NewOp | New operation from Java, it allocates space for a new object but it doesn't call a constructor |
|     | 26  | OP\_CONST\_STRING | Shuriken::MjolnIR::LoadString | Load a string to a register, the string must exists in the constant pool |
|     | 96  | OP\_SGET | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
|     | 97  | OP\_SGET\_WIDE | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
|     | 98  | OP\_SGET\_OBJECT | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
|     | 99  | OP\_SGET\_BOOLEAN | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
|     | 100 | OP\_SGET\_BYTE | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
|     | 101 | OP\_SGET\_CHAR | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
|     | 102 | OP\_SGET\_SHORT | Shuriken::MjolnIR::LoadFieldOp | Load a field value to a register, the field must exists in the class |
| Instruction21h | 21  | OP\_CONST\_HIGH16 | arith::ConstantFloatOp | Constant generation operation for a float, although not correct, most of the times this operation is used, in a source file, an OP\_CONST\_HIGH16 opcode is used |
|     | 25  | OP\_CONST\_WIDE\_HIGH16 | arith::ConstantIntOp | Constant generation operation for an int, although not correct, sometimes an integer is used, OP\_CONST\_WIDE\_HIGH16 opcode is generated |
| Instruction21s | 19  | OP\_CONST\_16 | arith::ConstantIntOp | Constant generation operation for an int, although not correct, sometimes an integer is used, OP\_CONST\_WIDE\_HIGH16 opcode is generated |
|     | 22  | OP\_CONST\_WIDE\_16 | arith::ConstantIntOp | Constant generation operation for an int, although not correct, sometimes an integer is used, OP\_CONST\_WIDE\_HIGH16 opcode is generated |
| Instruction21t | 56  | OP\_IF\_EQZ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate a comparison instruction and then a conditional branch operation |
|     | 57  | OP\_IF\_NEZ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate a comparison instruction and then a conditional branch operation |
|     | 58  | OP\_IF\_LTZ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate a comparison instruction and then a conditional branch operation |
|     | 59  | OP\_IF\_GEZ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate a comparison instruction and then a conditional branch operation |
|     | 60  | OP\_IF\_GTZ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate a comparison instruction and then a conditional branch operation |
|     | 61  | OP\_IF\_LEZ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate a comparison instruction and then a conditional branch operation |
| Instruction22b | 216 | OP\_ADD\_INT\_LIT8 | arith::ConstantIntOp + arith::AddIOp | Add operation with a constant value, we need to generate the constant and then the operation. |
|     | 217 | OP\_SUB\_INT\_LIT8 | arith::ConstantIntOp + arith::SubIOp | Sub operation with a constant value, we need to generate the constant and then the operation. |
|     | 218 | OP\_MUL\_INT\_LIT8 | arith::ConstantIntOp + arith::MulIOp | Mul operation with a constant value, we need to generate the constant and then the operation. |
|     | 219 | OP\_DIV\_INT\_LIT8 | arith::ConstantIntOp + arith::DivSIOp | Div operation with a constant value, we need to generate the constant and then the operation. |
|     | 220 | OP\_REM\_INT\_LIT8 | arith::ConstantIntOp + arith::RemSIOp | Rem operation with a constant value, we need to generate the constant and then the operation. |
|     | 221 | OP\_AND\_INT\_LIT8 | arith::ConstantIntOp + arith::AndIOp | And operation with a constant value, we need to generate the constant and then the operation. |
|     | 222 | OP\_OR\_INT\_LIT8 | arith::ConstantIntOp + arith::OrIOp | Or operation with a constant value, we need to generate the constant and then the operation. |
|     | 223 | OP\_XOR\_INT\_LIT8 | arith::ConstantIntOp + arith::XOrIOp | Xor operation with a constant value, we need to generate the constant and then the operation. |
|     | 224 | OP\_SHL\_INT\_LIT8 | arith::ConstantIntOp + arith::ShLIOp | Shift-Left operation with a constant value, we need to generate the constant and then the operation. |
|     | 225 | OP\_SHR\_INT\_LIT8 | arith::ConstantIntOp + arith::ShRSIOp | Shift-Right operation with a constant value, we need to generate the constant and then the operation. |
|     | 226 | OP\_USHR\_INT\_LIT8 | arith::ConstantIntOp + arith::ShRUIOp | Unsigned Shift-Right operation with a constant value, we need to generate the constant and then the operation. |
| Instruction22c | 82  | OP\_IGET | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 83  | OP\_IGET\_WIDE | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 84  | OP\_IGET\_OBJECT | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 85  | OP\_IGET\_BOOLEAN | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 86  | OP\_IGET\_BYTE | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 87  | OP\_IGET\_CHAR | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 88  | OP\_IGET\_SHORT | Shuriken::MjolnIR::LoadFieldOp | Load a value from a field, this field must exists in a class. |
|     | 89  | OP\_IPUT | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 90  | OP\_IPUT\_WIDE | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 91  | OP\_IPUT\_OBJECT | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 92  | OP\_IPUT\_BOOLEAN | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 93  | OP\_IPUT\_BYTE | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 94  | OP\_IPUT\_CHAR | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 95  | OP\_IPUT\_SHORT | Shuriken::MjolnIR::StoreFieldOp | Store a value in a field, this field must exists in a class. |
|     | 35  | OP\_NEW\_ARRAY | Shuriken::MjolnIR::NewArrayOp | Store a value in a field, this field must exists in a class. |
| Instruction22s | 208 | OP\_ADD\_INT\_LIT16 | arith::ConstantIntOp + arith::AddIOp | Add operation with a constant value, we need to generate the constant and then the operation. |
|     | 209 | OP\_SUB\_INT\_LIT16 | arith::ConstantIntOp + arith::SubIOp | Sub operation with a constant value, we need to generate the constant and then the operation. |
|     | 210 | OP\_MUL\_INT\_LIT16 | arith::ConstantIntOp + arith::MulIOp | Mul operation with a constant value, we need to generate the constant and then the operation. |
|     | 211 | OP\_DIV\_INT\_LIT16 | arith::ConstantIntOp + arith::DivSIOp | Div operation with a constant value, we need to generate the constant and then the operation. |
|     | 212 | OP\_REM\_INT\_LIT16 | arith::ConstantIntOp + arith::RemSIOp | Rem operation with a constant value, we need to generate the constant and then the operation. |
|     | 213 | OP\_AND\_INT\_LIT16 | arith::ConstantIntOp + arith::AndIOp | And operation with a constant value, we need to generate the constant and then the operation. |
|     | 214 | OP\_OR\_INT\_LIT16 | arith::ConstantIntOp + arith::OrIOp | Or operation with a constant value, we need to generate the constant and then the operation. |
|     | 215 | OP\_XOR\_INT\_LIT16 | arith::ConstantIntOp + arith::XOrIOp | Xor operation with a constant value, we need to generate the constant and then the operation. |
| Instruction22t | 50  | OP\_IF\_EQ | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate first a comparison and then a conditional branch. |
|     | 51  | OP\_IF\_NE | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate first a comparison and then a conditional branch. |
|     | 52  | OP\_IF\_LT | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate first a comparison and then a conditional branch. |
|     | 53  | OP\_IF\_GE | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate first a comparison and then a conditional branch. |
|     | 54  | OP\_IF\_GT | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate first a comparison and then a conditional branch. |
|     | 55  | OP\_IF\_LE | arith::CmpIOp + cf::CondBranchOp | Conditional jump, we need to generate first a comparison and then a conditional branch. |
| Instruction22x | 2   | OP\_MOVE\_FROM16 | Shuriken::MjolnIR::MoveOp | Move a value from one register to another. |
|     | 5   | OP\_MOVE\_WIDE\_FROM16 | Shuriken::MjolnIR::MoveOp | Move a value from one register to another. |
|     | 8   | OP\_MOVE\_OBJECT\_FROM16 | Shuriken::MjolnIR::MoveOp | Move a value from one register to another. |
| Instruction23x | 144 | OP\_ADD\_INT | arith::AddIOp | Add operation for integer values. |
|     | 155 | OP\_ADD\_LONG | arith::AddIOp | Add operation for long values. |
|     | 166 | OP\_ADD\_FLOAT | arith::AddFOp | Add operation for float values. |
|     | 171 | OP\_ADD\_DOUBLE | arith::AddFOp | Add operation for double values. |
|     | 145 | OP\_SUB\_INT | arith::SubIOp | Sub operation for integer values. |
|     | 156 | OP\_SUB\_LONG | arith::SubIOp | Sub operation for long values. |
|     | 167 | OP\_SUB\_FLOAT | arith::SubFOp | Sub operation for float values. |
|     | 172 | OP\_SUB\_DOUBLE | arith::SubFOp | Sub operation for double values. |
|     | 146 | OP\_MUL\_INT | arith::MulIOp | Mul operation for integer values. |
|     | 157 | OP\_MUL\_LONG | arith::MulIOp | Mul operation for long values. |
|     | 168 | OP\_MUL\_FLOAT | arith::MulFOp | Mul operation for float values. |
|     | 173 | OP\_MUL\_DOUBLE | arith::MulFOp | Mul operation for double values. |
|     | 147 | OP\_DIV\_INT | arith::DivSIOp | Div operation for integer values. |
|     | 158 | OP\_DIV\_LONG | arith::DivSIOp | Div operation for long values. |
|     | 169 | OP\_DIV\_FLOAT | arith::DivFOp | Div operation for float values. |
|     | 174 | OP\_DIV\_DOUBLE | arith::DivFOp | Div operation for double values. |
|     | 148 | OP\_REM\_INT | arith::RemSIOp | Rem operation for integer values. |
|     | 159 | OP\_REM\_LONG | arith::RemSIOp | Rem operation for long values. |
|     | 170 | OP\_REM\_FLOAT | arith::RemFOp | Rem operation for float values. |
|     | 175 | OP\_REM\_DOUBLE | arith::RemFOp | Rem operation for double values. |
|     | 149 | OP\_AND\_INT | arith::AndIOp | And operation for integer values. |
|     | 160 | OP\_AND\_LONG | arith::AndIOp | And operation for long values. |
|     | 150 | OP\_OR\_INT | arith::OrIOp | Or operation for integer values. |
|     | 161 | OP\_OR\_LONG | arith::OrIOp | Or operation for long values. |
|     | 151 | OP\_XOR\_INT | arith::XOrIOp | Xor operation for integer values. |
|     | 162 | OP\_XOR\_LONG | arith::XOrIOp | Xor operation for long values. |
|     | 152 | OP\_SHL\_INT | arith::ShLIOp | Shift-Left operation for integer values. |
|     | 163 | OP\_SHL\_LONG | arith::ShLIOp | Shift-Left operation for long values. |
|     | 153 | OP\_SHR\_INT | arith::ShRSIOp | Shift-Right operation for integer values. |
|     | 164 | OP\_SHR\_LONG | arith::ShRSIOp | Shift-Right operation for long values. |
|     | 154 | OP\_USHR\_INT | arith::ShRUIOp | Unsigned Shift-Right operation for integer values. |
|     | 165 | OP\_USHR\_LONG | arith::ShRUIOp | Unsigned Shift-Right operation for long values. |
| Instruction30t | 42  | OP\_GOTO\_32 | cf::BranchOp | Unconditional jump operation to an offset (positive or negative) |
| Instruction31c | 27  | OP\_CONST\_STRING\_JUMBO | Shuriken::MjolnIR::LoadString | Load a string to a register (string must exists in constant pool). |
| Instruction31i | 20  | OP\_CONST | arith::ConstantFloatOp | Create a constant float, although is not always the case, when using float values OP\_CONST opcode is used. |
|     | 23  | OP\_CONST\_WIDE\_32 | arith::ConstantFloatOp | Create a constant float, although is not always the case, when using float values OP\_CONST\_WIDE\_32 opcode is used. |
| Instruction32x | 3   | OP\_MOVE\_16 | Shuriken::MjolnIR::MoveOp | Move operation between registers. |
|     | 6   | OP\_MOVE\_WIDE\_16 | Shuriken::MjolnIR::MoveOp | Move operation between registers. |
|     | 9   | OP\_MOVE\_OBJECT\_16 | Shuriken::MjolnIR::MoveOp | Move operation between registers. |
| Instruction35c | 110 | OP\_INVOKE\_VIRTUAL | Shuriken::MjolnIR::InvokeOp | Invoke operation, this operation accepts parameters, and returns a value of a specified type. |
|     | 111 | OP\_INVOKE\_SUPER | Shuriken::MjolnIR::InvokeOp | Invoke operation, this operation accepts parameters, and returns a value of a specified type. |
|     | 112 | OP\_INVOKE\_DIRECT | Shuriken::MjolnIR::InvokeOp | Invoke operation, this operation accepts parameters, and returns a value of a specified type. |
|     | 113 | OP\_INVOKE\_STATIC | Shuriken::MjolnIR::InvokeOp | Invoke operation, this operation accepts parameters, and returns a value of a specified type. |
|     | 114 | OP\_INVOKE\_INTERFACE | Shuriken::MjolnIR::InvokeOp | Invoke operation, this operation accepts parameters, and returns a value of a specified type. |
| Instruction51l | 24  | OP\_CONST\_WIDE | arith::ConstantFloatOp | Create a constant float, although is not always the case, when using float values OP\_CONST\_WIDE opcode is used. |
