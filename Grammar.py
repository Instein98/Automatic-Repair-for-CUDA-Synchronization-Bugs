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
number = Word(nums+'.')
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = Literal("int")

# Arithmetic expressions
expression = Forward()

factor = Forward()
factor << (number | identifier | LParen + expression + RParen | unaryOp + factor)
term = factor + ZeroOrMore((MUL | DIV) + factor)
arithmeticExp = term + ZeroOrMore((ADD | SUB) + term)

_comparisonExp = arithmeticExp + ZeroOrMore((LE | GE | LT | GT) + arithmeticExp)
comparisonExp = _comparisonExp + ZeroOrMore((EQ | NE) + _comparisonExp)
_expression = comparisonExp + ZeroOrMore(AND + comparisonExp)
expression << _expression + ZeroOrMore(OR + _expression)  # include the arithExp and logicExp


statement = Literal("return") + space + expression + semicolon
function = dataType('returnType') + identifier('fnName') + LParen + RParen + LBrace + statement('statements') + RBrace
