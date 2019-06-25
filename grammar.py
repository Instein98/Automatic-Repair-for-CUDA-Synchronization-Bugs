"""
Program ::= Function
Function ::= "int" <name> "(" ")" "{" Statement "}"
Statement ::= "return" <space>+ Expression ";"
Expression ::= <int>
            |  UnaryOp Expression
            |  Expression BinOp Expression
UnaryOp ::= "!" | "~" | "-"
BinOp ::= "+" | "-" | "*" | "/"
"""

from pyparsing import *

LBrace, RBrace, LParen, RParen, semicolon = map(Suppress, "{}();")
unaryOp = oneOf("! - ~")  # must have space separator
binOp = oneOf("+ - * /")
number = Word(nums)
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = Literal("int")
expression = Forward()
expression << (number | unaryOp + expression | expression + binOp + expression)  # must have large parentheses
statement = Literal("return") + space + expression + semicolon
function = dataType('returnType') + identifier('fnName') + LParen + RParen + LBrace + statement('statements') + RBrace
