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
number = Word(nums+'.')
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = Literal("int")

arithmeticExp = Forward()
factor = Forward()
factor << (number | identifier | LParen + arithmeticExp + RParen | unaryOp + factor)
_term = Forward()
_term << Optional((MUL | DIV) + factor + _term)
term = factor + _term
_arithmeticExp = Forward()
_arithmeticExp << Optional((ADD | SUB) + term + _arithmeticExp)
arithmeticExp << term + _arithmeticExp

_comparisonExp = arithmeticExp + ZeroOrMore((LE | GE | LT | GT) + arithmeticExp)
comparisonExp = _comparisonExp + ZeroOrMore((EQ | NE) + _comparisonExp)
_expression = comparisonExp + ZeroOrMore(AND + comparisonExp)
expression = _expression + ZeroOrMore(OR + _expression)  # include the arithmeticExp and logicExp

"""Failed precedence_2.c"""

statement = Literal("return") + space + expression + semicolon
function = dataType('returnType') + identifier('fnName') + LParen + RParen + LBrace + statement('statements') + RBrace
