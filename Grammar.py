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
    # print(toks)
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
    elif t == '&':
        exprStack.append('preUnary &')
    elif t == '*':
        exprStack.append('preUnary *')

    t = toks[-1]
    if t == '--':
        exprStack.append('postUnary --')
    elif t == '++':
        exprStack.append('postUnary ++')

def pushFunc(strg, loc, toks):
    funcInfo = toks[0].asList()
    arguNum = 0
    for x in funcInfo:
        if x == ',':
            arguNum += 1
    if arguNum == 0:
        if len(funcInfo) > 1:
            arguNum += 1
    else:
        arguNum += 1
    dict = {'funcName': funcInfo[0], 'arguNum': arguNum}
    exprStack.append(dict)

# def pushIdent(strg, loc, toks):
#     # exprStack.append(toks)
#     print(toks)
#     print('')

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
preUnaryOp = oneOf("! - ~ ++ -- * &")  # must have space separator
postUnaryOp = oneOf("++ --")

number = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
identifier = Word(alphas+'_', alphanums+'_')
space = White(min=1)
dataType = (Word(alphas+'_', alphanums+'_') + Literal('*') | Word(alphas+'_', alphanums+'_'))

# Arithmetic expressions
expression = Forward()
funcCall = Forward()
atom = ((0,None) * preUnaryOp + (Group(funcCall).setParseAction(pushFunc) | (Literal("false") | Literal("true") |
        number | identifier | QuotedString('"')).setParseAction(pushFirst)| Group(LParen + expression + RParen)) +
        (0,None) * postUnaryOp).setParseAction(pushUnary)
_atom = atom + ZeroOrMore((Literal('[') + Optional(expression) + Literal(']')).setParseAction(pushFirst))
factor = _atom + ZeroOrMore(((Literal('.') | Literal('->')) + _atom).setParseAction(pushFirst))
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
comment = Literal('//') + restOfLine | nestedExpr("/**", "*/")
statement = Forward()  # return, declare, assign, if,
statementBlock = (statement | LBrace + ZeroOrMore(statement) + RBrace)
ifPart = Literal("if") + LParen + expression('condExp') + RParen + statementBlock('IfBlock')
ifPart.ignore(comment)
elsePart = Optional(Literal("else") + statementBlock('ElseBlock'))
elsePart.ignore(comment)
declaration = Optional(Literal('const') | Word('__', alphanums+'_')) + dataType('dataType') + space + \
              ((_atom | identifier)('varName')) + Optional(ASSIGN + expression('initialValue'))
assignment = expression('left') + ASSIGN + expression('right')
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
            | Literal('#') + restOfLine('Single')
            )
statement.ignore(comment)

# Function
argList = expression + ZeroOrMore(Literal(',') + expression)  # exp, exp, exp ...
initArgList = delimitedList(Optional(Literal('const')) + dataType + Optional(Literal('const')) + identifier)
funcDeclare = Optional(Literal('template') + Literal('<') + delimitedList((dataType | Literal('typename')) +
              identifier) + Literal('>')) + Word('__', alphanums+'_') + Optional(Literal('static')) + dataType('returnType') + identifier('fnName') + \
              LParen + Optional(initArgList)('initArgs') + RParen + LBrace + ZeroOrMore(statement) + RBrace
funcDeclare.ignore(comment)
funcCall << identifier('fnName') + Optional(Literal('<') + dataType + Literal('>'))('T') + LParen + \
            Optional(argList)('arguList') + RParen
