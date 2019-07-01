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

exprStack = []

def pushFirst(strg, loc, toks):
    exprStack.append(toks[0])

def pushUnary(strg, loc, toks):
    for t in toks:
        if t == '-':
            exprStack.append('unary -')
        elif t == '!':
            exprStack.append('unary !')
        elif t == '~':
            exprStack.append('unary ~')
        else:
            break

def parseExp(exp):
    """Input a expression string, output the postfix token list"""
    global exprStack
    exprStack[:] = []
    try:
        expression.parseString(exp, parseAll=True)
    except ParseException as pe:
        print(exp, "Parsing Failed:", str(pe))
    except Exception as e:
        print(exp, "Failed:", str(e))
    else:
        return exprStack

def expToString(expList): # The expression list in parse result of statement
    result = ''.join(str(e) for e in expList)
    result = result.replace('\'','').replace(',',' ').replace('[','(').replace(']',')')
    return result


LBrace, RBrace, LParen, RParen, comma= map(Suppress, "{}(),")
semicolon = Literal(";")
ADD, SUB, MUL, DIV = map(Literal, "+-*/")
AND, OR, EQ, NE, LT, LE, GT, GE, ASSIGN = map(Literal, ["&&", "||", "==", "!=", "<", "<=", ">", ">=", "="])
unaryOp = oneOf("! - ~")  # must have space separator
number = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = Literal("int")

# Arithmetic expressions
expression = Forward()
atom = ((0,None)*unaryOp + ((number | identifier + LParen + expression + RParen | identifier).setParseAction(pushFirst)
    | Group(LParen + expression + RParen))).setParseAction(pushUnary)
factor = atom
term = factor + ZeroOrMore(((MUL | DIV) + factor).setParseAction(pushFirst))
arithmeticExp = term + ZeroOrMore(((ADD | SUB) + term).setParseAction(pushFirst))

_comparisonExp = arithmeticExp + ZeroOrMore(((LE | GE | LT | GT) + arithmeticExp).setParseAction(pushFirst))
comparisonExp = _comparisonExp + ZeroOrMore(((EQ | NE) + _comparisonExp).setParseAction(pushFirst))
_expression = comparisonExp + ZeroOrMore((AND + comparisonExp).setParseAction(pushFirst))
expression << _expression + ZeroOrMore((OR + _expression).setParseAction(pushFirst))



statement = ((Literal("return") + space + expression('retExp') + semicolon)('Return')
            | (dataType('dataType') + space + identifier('varName') +
               Optional(ASSIGN + expression('initialValue')) + semicolon)('Declaration')  # Declaration [Initialization]
            | (identifier('left') + ASSIGN + expression('right') + semicolon)('Assignment'))  # Assignment
function = dataType('returnType') + identifier('fnName') + LParen + \
           RParen + LBrace + ZeroOrMore(statement) + RBrace
