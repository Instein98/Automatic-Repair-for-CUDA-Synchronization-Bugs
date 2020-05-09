from Grammar import *
from abc import ABC, abstractmethod
from Parser import *


unaryOperator = ['preUnary --', 'preUnary ++', 'preUnary ~', 'preUnary !', 'preUnary -', 'preUnary &',
                 'preUnary *', 'postUnary --', 'postUnary ++']
unaryLiteral = ['--', '++', '~', '!', '-', '&', '*', '--', '++']
binaryOperator = ['+', '-', '*', '/', '>', '>=', '<', '<=', '==', '!=', '&&', '||', '%', '<<', '>>',
                  '/=', '*=', '%=', '+=', '-=', '<<=', '>>=', '&=', '^=', '|=', '.', '->']
ternaryOperator = ['?']


def fail(str = "!!!!!!! PARSE FAILED !!!!!!!!"):
    print(str)
    exit(0)


def findField(content, start, end):
    temp = content.find(start)
    if temp == -1:
        return None
    st = [temp]
    contP = temp + 1
    lastEnd = None
    while len(st) != 0:
        if content[contP] == start:
            st.append(contP)
        elif content[contP] == end:
            lastEnd = contP
            st.remove(st[-1])
        contP += 1
    return lastEnd


def parseStatement(content):
    if content == ';':
        return None
    parseResult = statement.parseString(content)
    parseResultDict = parseResult.asDict()
    if 'If' in parseResultDict:
        if parseResult.asList()[0] == 'if':
            return If(content, parseResult)
    if 'For' in parseResultDict:
        if parseResult.asList()[0] == 'for':
            return For(content, parseResult)
    if 'While' in parseResultDict:
        if parseResult.asList()[0] == 'while':
            return While(content, parseResult)
    if 'DoWhile' in parseResultDict:
        if parseResult.asList()[0] == 'do':
            return DoWhile(content, parseResult)
    if 'Return' in parseResultDict:
        return Return(content, parseResult)
    if 'Declaration' in parseResultDict:
        return Declaration(content, parseResult)
    if 'Assignment' in parseResultDict:
        return Assignment(content, parseResult)
    if 'Single' in parseResultDict:
        return Single(content, parseResult)
    if 'Exp' in parseResultDict:
        return Expression(content.replace(';', ''))

    return None


def outputNode(node):
    if type(node) == str:
        return node
    else:
        return node.output()


def getStatementListOutput(sList, currentIndentLevel):
    res = ''
    for statement in sList:
        if type(statement) == Expression:
            expr = statement.output()[1:-1]  # removing ( and )
            res += (currentIndentLevel + 1) * '\t' + expr + ';\n'
        else:
            res += statement.output(currentIndentLevel + 1)  # assume output of this include ; and \n
    return res



class ASTNode(ABC):
    def __init__(self, content):
        self.content = content  # origin text of the node

    def __str__(self):
        pass

    @abstractmethod
    def output(self, indentLevel=0):
        pass

    __repr__ = __str__


class Program(ASTNode):
    def __init__(self, path):
        self.path = path
        self.functionList = []
        f = open(self.path, 'r')
        self.content = f.read()
        f.close()
        self.parseFunctions()

    def parseFunctions(self):
        scanResult = funcDeclare.scanString(self.content)
        self.content = self.content.expandtabs()
        for x in scanResult:
            self.functionList.append(FunctionDeclare(self.content[x[1]:x[2]]))
        if not self.functionList:
            fail()


class FunctionDeclare(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.argList = []
        self.parseFunction()
        self.getArgs()

    def parseFunction(self):
        parseResult = funcDeclare.parseString(self.content).asDict()
        self.returnType = parseResult['returnType']
        self.fnName = parseResult['fnName'][0]
        self.statementList = []
        scanResult = statement.scanString(self.content)
        self.content = self.content.expandtabs()  # To fix the wrong start & end location
        for x in scanResult:  # the content of each statement
            state = parseStatement(self.content[x[1]:x[2]])
            if state != None:
                self.statementList.append(state)
        if not self.statementList:
            fail()

    def getArgs(self):
        argInfo = self.content[self.content.find('(')+1:self.content.find(')')]
        if not argInfo.strip():
            return
        argInfo = argInfo.split(',')
        for arg in argInfo:
            argInfo = arg.rsplit(' ',1)
            self.argList.append((argInfo[0].strip(), argInfo[1].strip()))

    def output(self, indentLevel=0):
        res = indentLevel*'\t' + self.content[:self.content.find('{')+1] + '\n'
        # for statement in self.statementList:
        #     res += statement.output(indentLevel+1)
        res += getStatementListOutput(self.statementList, indentLevel)
        res += indentLevel*'\t' + '}\n'
        return res


class Statement(ASTNode, ABC):
    def __init__(self, content, parseResult):
        super().__init__(content)
        if parseResult is not None:
            self.parseResult = parseResult
            self.parseResultDict = parseResult.asDict()
            self.parse()

    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def output(self, indentLevel=0):
        pass


class Assignment(Statement):
    def parse(self):
        self.leftSide = Expression(self.content[:self.content.find('=')])
        self.rightSide = Expression(self.content[self.content.find('=')+1:].replace(';',''))

    def output(self, indentLevel):
        res = indentLevel*'\t' + self.leftSide.output() if type(self.leftSide) != str else self.leftSide
        res += " = "
        res += self.rightSide.output() + ';\n' if type(self.leftSide) != str else self.leftSide + ';\n'
        return res


class Return(Statement):
    def parse(self):
        returnTarget = self.content[self.content.find('return')+6:].replace(';', '')
        if len(returnTarget.strip()) == 0:
            self.returnExp = Single('return', None)
        else:
            self.returnExp = Expression(returnTarget)

    def output(self, indentLevel=0):
        if type(self.returnExp) == Single:
            return indentLevel*'\t' + self.returnExp.output()
        else:
            return indentLevel*'\t' + self.returnExp.output() + ';\n'


class Declaration(Statement):
    def parse(self):
        self.declareList = None
        self.initExp = None
        if ',' not in self.content:
            self.dataType = self.parseResultDict['dataType']
            self.dataType = self.content[:self.content.find(self.dataType) + len(self.dataType)]
            self.varName = self.parseResultDict['varName']
            if 'initialValue' in self.parseResultDict:
                self.initExp = Expression(self.content[self.content.find('=') + 1:].replace(';',''))
        else:
            splitList = self.content.split(",")
            self.declareList = []
            for i in range(len(splitList)):
                if i == 0:
                    content = splitList[i] + ';'
                    parseResult = statement.parseString(content)
                    self.declareList.append(Declaration(content, parseResult))
                else:
                    content = self.declareList[0].dataType + ' ' + splitList[i] + '' if i == len(splitList)-1 else ';'
                    parseResult = statement.parseString(content)
                    self.declareList.append(Declaration(content, parseResult))

    def output(self, indentLevel=0):
        # __shared__ Real sbuf[TileDim][TileDim + 1];
        if "__shared__" in self.content:
            return indentLevel*'\t' + self.content + '\n'
        res = indentLevel*'\t'
        if ',' not in self.content:
            res += self.dataType + ' '
            res += self.varName + ' '
            if self.initExp is not None:
                res += ' = ' + self.initExp.output()
            res += ';\n'
        return res


class If(Statement):
    def parse(self):
        self.ifStatementList = []
        self.elseStatementList = []

        ifBound = list(ifPart.scanString(self.content))[0]
        ifContent = self.content[ifBound[1]:ifBound[2]]
        ifTemp = findField(ifContent, '(', ')')
        self.condition = Expression(ifContent[ifContent.find('(')+1:ifTemp])
        ifExclude = list((Literal("if") + LParen + expression('condExp') + RParen).scanString(ifContent))[0]
        ifContent = ifContent[ifExclude[2]+1:]
        elseContent = self.content[ifBound[2]+1:]
        elseContent = elseContent[elseContent.find("else") + 4:]

        for x in statement.scanString(ifContent):
            state = parseStatement(ifContent[x[1]:x[2]])
            if state != None:
                self.ifStatementList.append(state)
        for x in statement.scanString(elseContent):
            state = parseStatement(elseContent[x[1]:x[2]])
            if state != None:
                self.elseStatementList.append(state)

    def output(self, indentLevel=0):
        indent = indentLevel*'\t'
        res = indent + 'if (' + self.condition.output() + '){\n'
        for statement in self.ifStatementList:
            res += statement.output(indentLevel+1)
        if len(self.elseStatementList) == 0:
            res += indent + '}\n'
        else:
            res += indent + '} else {\n'
            res += getStatementListOutput(self.elseStatementList, indentLevel)
            # for statement in self.elseStatementList:
            #     res += statement.output(indentLevel+1)
            res += indent + '}\n'
        return res


# todo for(:) ?
class For(Statement):
    def parse(self):
        self.statementList = []
        self.initialization = None
        self.loopCondition = None
        self.postStatement = None
        leftParenPos = self.content.find('(')
        rightParenPos = self.content.find(')')
        firstSemiPos = self.content.find(';')
        secondSemiPos = self.content.find(';', firstSemiPos + 1)
        if 'init' in self.parseResultDict:
            self.initialization = parseStatement(self.content[leftParenPos+1:firstSemiPos]+';')
        else:
            self.initialization = None
        if 'cond' in self.parseResultDict:
            self.loopCondition = parseStatement(self.content[firstSemiPos+1:secondSemiPos]+';')
        else:
            self.loopCondition = None
        if 'post' in self.parseResultDict:
            self.postStatement = parseStatement(self.content[secondSemiPos+1:rightParenPos]+';')
        else:
            self.postStatement = None
        forExclude = (Literal("for") + LParen + Optional((declaration | assignment)) + semicolon +
              Optional(expression) + semicolon + Optional((assignment | expression)) + RParen)
        forExclude = list(forExclude.scanString(self.content))[0]
        stateContent = self.content[forExclude[2]+1:]

        for x in statement.scanString(stateContent):
            state = parseStatement(stateContent[x[1]:x[2]])
            if state != None:
                self.statementList.append(state)

    def output(self, indentLevel=0):
        indent = indentLevel*'\t'
        res = indent + 'for ('
        if self.initialization is not None:
            if isinstance(self.initialization, Statement):
                res += self.initialization.output(0).replace('\n', '')
            elif type(self.initialization) == Expression:
                res += self.initialization.output(0)[1:-1].strip() + ';'  # remove '(', ')'
        if self.loopCondition is not None and type(self.loopCondition) == Expression:
            res += self.loopCondition.output(0)[1:-1] + ';'
        if self.postStatement is not None:
            if type(self.postStatement) == Statement:
                output = self.postStatement.output(0)
                res += output[:output.find(';')]  # remove ; and \n
            elif type(self.postStatement) == Expression:
                res += self.postStatement.output(0)[1:-1]
        res += '){\n'
        res += getStatementListOutput(self.statementList, indentLevel)
        # for statement in self.statementList:
        #     res += statement.output(indentLevel+1)
        res += indent + '}\n'
        return res


class While(Statement):
    def parse(self):
        self.statementList = []
        whileExclude = list((Literal("while") + LParen + expression + RParen).scanString(self.content))[0]
        whileContent = self.content[:whileExclude[2]+1]
        self.loopCondition = Expression(whileContent[whileContent.find('(')+1 : whileContent.find(')')])
        stateContent = self.content[whileExclude[2]+1:]
        for x in statement.scanString(stateContent):
            state = parseStatement(stateContent[x[1]:x[2]])
            if state != None:
                self.statementList.append(state)

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        res = indent + 'while ('
        res += self.loopCondition.output()
        res += '){\n'
        res += getStatementListOutput(self.statementList, indentLevel)
        # for statement in self.statementList:
        #     res += statement.output(indentLevel+1)
        res += indent + '}\n'
        return res


class DoWhile(Statement):
    def parse(self):
        self.statementList = []
        whileExclude = list((Literal("while") + LParen + expression + RParen).scanString(self.content))[0]
        whileContent = self.content[whileExclude[1]:]  # the while part
        self.loopCondition = parseStatement(whileContent[whileContent.find('(')+1:whileContent.find(')')]+';')
        stateContent = self.content[:whileExclude[1]]
        for x in statement.scanString(stateContent):
            state = parseStatement(stateContent[x[1]:x[2]])
            if state != None:
                self.statementList.append(state)

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        res = indent + 'do {\n'
        res += getStatementListOutput(self.statementList, indentLevel)
        # for statement in self.statementList:
        #     res += statement.output(indentLevel+1)
        res += indent + '} while ('
        res += self.loopCondition.output() if type(self.loopCondition) == Expression else '???'
        res += ');\n'


class Single(Statement):
    """break/continue etc."""
    def parse(self):
        pass

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + self.content + '\n'


class Expression(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.childNode = None
        self.parseExpression()

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        if type(self.childNode) == str:
            return self.childNode
        return indent + self.childNode.output()

    def parseExpression(self):
        postfixTokens = parseExp(self.content).copy()
        # print(self.content," -> ",postfixTokens)
        self.constructExpNode(postfixTokens)

    def constructExpNode(self, postfixTokens):
        index = 0
        while len(postfixTokens) > 1:
            # Meet unary operator, pop
            if postfixTokens[index] in unaryOperator:
                if index - 1 < 0:
                    fail("Failed to construct AST for expression!: Unary Operator")
                Uop = UnaryOp(postfixTokens[index], postfixTokens[index-1])
                postfixTokens[index-1] = Uop
                del postfixTokens[index]
                continue

            # Meet binary operator, pop
            elif postfixTokens[index] in binaryOperator:
                if index - 2 < 0:
                    fail("Failed to construct AST for expression!: Binary Operator")
                Bop = BinOp(postfixTokens[index], postfixTokens[index-2], postfixTokens[index-1])
                postfixTokens[index-2] = Bop
                del postfixTokens[index]
                del postfixTokens[index-1]
                index -= 1
                continue

            # Meet ternary operator, pop
            elif postfixTokens[index] in ternaryOperator:
                if index - 3 < 0:
                    fail("Failed to construct AST for expression!: Binary Operator")
                Top = TernaryOp(postfixTokens[index], postfixTokens[index-3], postfixTokens[index-2], postfixTokens[index-1])
                postfixTokens[index - 3] = Top
                del postfixTokens[index]
                del postfixTokens[index - 1]
                del postfixTokens[index - 2]
                index -= 2
                continue

            # Meet function call
            elif isinstance(postfixTokens[index], dict):
                funcInfo = postfixTokens[index]
                funcName = funcInfo['funcName']
                arguNum = funcInfo['arguNum']
                if index - arguNum < 0:
                    fail("Failed to construct AST for expression!: Function Call")
                tempScan = list((Literal(funcName) + Literal('<')).scanString(self.content))
                if tempScan:
                    funcName = self.content[:self.content.find('>')+1]
                FunCall = FunctionCall(self.content, funcName, postfixTokens[index - arguNum:index])
                postfixTokens[index-arguNum] = FunCall
                for i in range(0, arguNum):
                    del postfixTokens[index-i]
                index -= (arguNum - 1)
                continue

            elif postfixTokens[index] == '[':
                if index - 2 < 0:
                    fail("Failed to construct AST for expression!: Array")
                Arr = Array(postfixTokens[index - 2], postfixTokens[index - 1])
                postfixTokens[index - 2] = Arr
                del postfixTokens[index]
                del postfixTokens[index - 1]
                index -= 1
                continue

            # Meet identifiers, just push
            else:
                index += 1
                if index > len(postfixTokens) - 1:
                    fail("Failed to construct AST for expression!: Index out of range")

        if isinstance(postfixTokens[0], dict):
            funcInfo = postfixTokens[index]
            funcName = funcInfo['funcName']
            arguList = []
            # todo how to get function content?
            postfixTokens[0] = FunctionCall(None, funcName, arguList)

        self.childNode = postfixTokens[0]


class Array(ASTNode):
    def __init__(self, arr, index):
        super().__init__('Array')
        self.arr = arr
        self.index = index

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + outputNode(self.arr) + '[' + outputNode(self.index) + ']'


class FunctionCall(ASTNode):
    def __init__(self, content, funcName, arguList):
        super().__init__(content)
        self.funcName = funcName
        self.arguList = arguList
        # content = funcName + str(arguList).replace('[', '(').replace(']', ')')
        # super().__init__('content')

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + self.content


class UnaryOp(ASTNode):
    def __init__(self, op, opr):
        super().__init__(op)
        self.operator = op
        self.operand = opr

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        for i, operator in enumerate(unaryOperator):
            if self.operator == operator:
                if 'pre' in operator:
                    return indent + unaryLiteral[i] + self.operand
                elif 'post' in operator:
                    return indent + self.operand + unaryLiteral[i]


class BinOp(ASTNode):
    def __init__(self, op, left, right):
        super().__init__(op)
        self.op = op
        self.left = left
        self.right = right

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        res = indent
        if self.op != '.':  # do not need ( for a.b
            res += '('
        res += self.left if type(self.left) == str else self.left.output()
        res += self.op
        res += self.right if type(self.right) == str else self.right.output()
        if self.op != '.':  # do not need () for a.b
            res += ')'
        return res


class TernaryOp(ASTNode):
    """conditional expression: a ? b : c"""
    def __init__(self, op, cond, yes, no):
        super().__init__(op)
        self.op = op
        self.cond = cond
        self.yes = yes
        self.no = no

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + self.content


def kernelAST(path, funcName):
    P = Program(path)  # Build AST with P as the root node
    for function in P.functionList:
        if function.fnName == funcName:
            return function
    print('No such kernel function matched!')
    return None


if __name__ == "__main__":
    ParserElement.enablePackrat()
    # fnName = "_sum_reduce"  # OK
    # fnName = "_copy_low_upp"  # roughly OK, "return;" as expression
    fnName = "_copy_from_mat_trans"
    sourcePath = "D:/DATA/Python_ws/CUDA_Parser/test.cpp"
    parser = Parser()
    with open(sourcePath) as source:
        content = source.read()
        fnContent = parser.getFunctionContent(fnName, content)
    func = FunctionDeclare(fnContent)
    print(func.output())
