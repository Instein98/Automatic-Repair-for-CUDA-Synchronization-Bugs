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
ADD, SUB, MUL, DIV = map(Literal, "+-*/")
AND, OR, EQ, NE, LT, LE, GT, GE = map(Literal, ["&&", "||", "==", "!=", "<", "<=", ">", ">="])
unaryOp = oneOf("! - ~")  # must have space separator
# binOp = oneOf("+ - * /")
number = Word(nums)
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = Literal("int")

expression = Forward()
factor = Forward()
factor << (number | identifier | LParen + expression + RParen | unaryOp + factor)
_term = Forward()
_term << Optional((MUL | DIV) + factor + _term)
term = factor + _term
_expression = Forward()
_expression << Optional((ADD | SUB) + term + _expression)
expression << term + _expression

statement = Literal("return") + space + expression + semicolon
function = dataType('returnType') + identifier('fnName') + LParen + RParen + LBrace + statement('statements') + RBrace
