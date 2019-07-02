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
    t = toks[0]
    if t == '--':
        exprStack.append('preUnary --')
    elif t == '++':
        exprStack.append('preUnary ++')
    elif t == '-':
        exprStack.append('preUnary -')
    elif t == '!':
        exprStack.append('preUnary !')
    elif t == '~':
        exprStack.append('preUnary ~')

    t = toks[-1]
    if t == '--':
        exprStack.append('postUnary --')
    elif t == '++':
        exprStack.append('postUnary ++')


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

# Newly added
BinEQ = oneOf("/= *= %= += -= <<= >>= &= ^= |=")
MOD, LS, RS  = map(Literal, ["%", "<<", ">>"])
preUnaryOp = oneOf("! - ~ ++ --")  # must have space separator
postUnaryOp = oneOf("++ --")

number = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = Literal("int")

# Arithmetic expressions
expression = Forward()
atom = ((0,None) * preUnaryOp + ((Literal("false") | Literal("true") | number | identifier + LParen + expression + RParen
    | identifier).setParseAction(pushFirst) | Group(LParen + expression + RParen)) + (0,None) * postUnaryOp).setParseAction(pushUnary)
factor = atom
term = factor + ZeroOrMore(((MUL | DIV | MOD) + factor).setParseAction(pushFirst))
arithmeticExp = term + ZeroOrMore(((ADD | SUB) + term).setParseAction(pushFirst))
shiftExp = arithmeticExp + ZeroOrMore(((LS | RS) + arithmeticExp).setParseAction(pushFirst))

# Logical expression
_comparisonExp = shiftExp + ZeroOrMore(((LE | GE | LT | GT) + shiftExp).setParseAction(pushFirst))
comparisonExp = _comparisonExp + ZeroOrMore(((EQ | NE) + _comparisonExp).setParseAction(pushFirst))
ANDexpression = comparisonExp + ZeroOrMore((AND + comparisonExp).setParseAction(pushFirst))
ORexpression = ANDexpression + ZeroOrMore((OR + ANDexpression).setParseAction(pushFirst))

# Conditional expression
conditionalExp = ORexpression + Optional((Literal("?") + ORexpression + Literal(":") + ORexpression).setParseAction(pushFirst))
expression << conditionalExp + ZeroOrMore((BinEQ + conditionalExp).setParseAction(pushFirst))

# Statements
statement = Forward()  # return, declare, assign, if,
statementBlock = (statement | LBrace + ZeroOrMore(statement) + RBrace)
ifPart = Literal("if") + LParen + expression('condExp') + RParen + statementBlock('IfBlock')
elsePart = Optional(Literal("else") + statementBlock('ElseBlock'))
declaration = dataType('dataType') + space + identifier('varName') + Optional(ASSIGN + expression('initialValue'))
assignment = identifier('left') + ASSIGN + expression('right')
statement << ((Literal("return") + space + expression('retExp') + semicolon)('Return')  # return + exp
            | (declaration + semicolon)('Declaration')  # Declaration [Initialization]
            | (assignment + semicolon)('Assignment')  # Assignment
            | (ifPart + elsePart)('If')  # If statement
            | (Literal("for") + LParen + Optional((declaration | assignment)('init')) +
              semicolon + Optional(expression('cond')) + semicolon +
              Optional((assignment | expression)('post')) +
              RParen + statementBlock)('For')  # For statement
            | (Literal("while") + LParen + expression('cond') + RParen + statementBlock)('While')
            | (Literal("do") + LBrace + ZeroOrMore(statement) + RBrace + Literal("while") +
               LParen + expression('cond') + RParen + semicolon)('DoWhile')
            | (oneOf("break continue __syncthreads()") + semicolon)('Single')
            | (Optional(expression) + semicolon)('Exp')
            )

# Function
function = dataType('returnType') + identifier('fnName') + LParen + \
           RParen + LBrace + ZeroOrMore(statement) + RBrace
