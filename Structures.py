from Grammar import *
from abc import ABC, abstractmethod
from Parser import *


unaryOperator = ['preUnary --', 'preUnary ++', 'preUnary ~', 'preUnary !', 'preUnary -', 'preUnary &',
                 'preUnary *', 'postUnary --', 'postUnary ++']
unaryLiteral = ['--', '++', '~', '!', '-', '&', '*', '--', '++']
binaryOperator = ['+', '-', '*', '/', '>', '>=', '<', '<=', '==', '!=', '&&', '||', '%', '<<', '>>',
                  '/=', '*=', '%=', '+=', '-=', '<<=', '>>=', '&=', '^=', '|=', '.', '->']
ternaryOperator = ['?']


def fail(msg="!!!!!!! PARSE FAILED !!!!!!!!"):
    print(msg)
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


def parseStatement(content, baseLineNo, parent):
    if content == ';':
        return None
    parseResult = statement.parseString(content)
    parseResultDict = parseResult.asDict()
    if 'If' in parseResultDict:
        if parseResult.asList()[0] == 'if':
            return If(content, parseResult, baseLineNo, parent)
    if 'For' in parseResultDict:
        if parseResult.asList()[0] == 'for':
            return For(content, parseResult, baseLineNo, parent)
    if 'While' in parseResultDict:
        if parseResult.asList()[0] == 'while':
            return While(content, parseResult, baseLineNo, parent)
    if 'DoWhile' in parseResultDict:
        if parseResult.asList()[0] == 'do':
            return DoWhile(content, parseResult, baseLineNo, parent)
    if 'Return' in parseResultDict:
        return Return(content, parseResult, baseLineNo, parent)
    if 'Declaration' in parseResultDict:
        return Declaration(content, parseResult, baseLineNo, parent)
    if 'Assignment' in parseResultDict:
        return Assignment(content, parseResult, baseLineNo, parent)
    if 'Single' in parseResultDict:
        return Single(content, parseResult, baseLineNo, parent)
    if 'Exp' in parseResultDict:
        return Expression(content.replace(';', ''), baseLineNo, parent)

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
    def __init__(self, content, baseLineNo, parent):
        self.content = content  # origin text of the node
        self.baseLineNo = baseLineNo  # the line number of the node
        self.parent = parent
        self.children = None

    def __str__(self):
        pass

    @abstractmethod
    def output(self, indentLevel=0):
        pass

    __repr__ = __str__


# class Program(ASTNode):
#     def __init__(self, path):
#         self.path = path
#         self.functionList = []
#         f = open(self.path, 'r')
#         self.content = f.read()
#         f.close()
#         self.parseFunctions()
#
#     def parseFunctions(self):
#         scanResult = funcDeclare.scanString(self.content)
#         self.content = self.content.expandtabs()
#         for x in scanResult:
#             self.functionList.append(FunctionDeclare(self.content[x[1]:x[2]], self.baseLineNo))
#         if not self.functionList:
#             fail()


class FunctionDeclare(ASTNode):
    def __init__(self, content, baseLineNo, parent):
        super().__init__(content, baseLineNo, parent)
        self.argList = []
        self.parseFunction()
        self.children = self.statementList
        self.getArgs()

    def parseFunction(self):
        parseResult = funcDeclare.parseString(self.content).asDict()
        self.returnType = parseResult['returnType']
        self.fnName = parseResult['fnName'][0]
        self.statementList = []
        scanResult = statement.scanString(self.content)
        self.content = self.content.expandtabs()  # To fix the wrong start & end location
        previousStmt = None
        for x in scanResult:  # the content of each statement
            relativeLineNo = getLineNoByPos(self.content, x[1])
            state = parseStatement(self.content[x[1]:x[2]], relativeLineNo + self.baseLineNo, self)
            if state is not None:
                state.previous = previousStmt
                if previousStmt is not None:
                    previousStmt.next = state
                self.statementList.append(state)
                previousStmt = state
        if not self.statementList:
            fail()

    def getArgs(self):
        argContent = self.content[self.content.find('(')+1:self.content.find(')')]
        if not argContent.strip():
            return
        argContent = argContent.split(',')
        for arg in argContent:
            if '*' in arg:
                idx = arg.rindex('*')
                self.argList.append((arg[:idx+1].strip(), arg[idx+1:].strip()))
            else:
                argContent = arg.rsplit(' ', 1)
                self.argList.append((argContent[0].strip(), argContent[1].strip()))

    def output(self, indentLevel=0):
        res = indentLevel*'\t' + self.content[:self.content.find('{')+1] + '\n'
        # for statement in self.statementList:
        #     res += statement.output(indentLevel+1)
        res += getStatementListOutput(self.statementList, indentLevel)
        res += indentLevel*'\t' + '}\n'
        return res


class Statement(ASTNode, ABC):
    def __init__(self, content, parseResult, baseLineNo, parent, previous=None, next=None):
        super().__init__(content, baseLineNo, parent)
        self.previous = previous
        self.next = next
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
        self.leftSide = Expression(self.content[:self.content.find('=')], self.baseLineNo, self)
        self.rightSide = Expression(self.content[self.content.find('=')+1:].replace(';', ''), self.baseLineNo, self)

    def output(self, indentLevel=0):
        res = indentLevel*'\t' + self.leftSide.output() if type(self.leftSide) != str else self.leftSide
        res += " = "
        res += self.rightSide.output() + ';\n' if type(self.leftSide) != str else self.leftSide + ';\n'
        return res


class Return(Statement):
    def parse(self):
        returnTarget = self.content[self.content.find('return')+6:].replace(';', '')
        if len(returnTarget.strip()) == 0:
            self.returnExp = Single('return;', None, self.baseLineNo, self)
        else:
            # todo return should not be expression?? a+=3
            self.returnExp = Expression(returnTarget, self.baseLineNo, self)

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
                self.initExp = Expression(self.content[self.content.find('=') + 1:].replace(';', ''), self.baseLineNo, self)
        else:
            splitList = self.content.split(",")
            self.declareList = []
            for i in range(len(splitList)):
                if i == 0:
                    content = splitList[i] + ';'
                    parseResult = statement.parseString(content)
                    self.declareList.append(Declaration(content, parseResult, self.baseLineNo, self))
                else:
                    content = self.declareList[0].dataType + ' ' + splitList[i] + '' if i == len(splitList)-1 else ';'
                    parseResult = statement.parseString(content)
                    self.declareList.append(Declaration(content, parseResult, self.baseLineNo, self))

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
        self.condition = Expression(ifContent[ifContent.find('(')+1:ifTemp], self.baseLineNo, self)
        ifExclude = list((Literal("if") + LParen + expression('condExp') + RParen).scanString(ifContent))[0]
        ifContent = ifContent[ifExclude[2]+1:]
        ifContentBaseLineNo = getLineNoByPos(self.content, self.content.index(ifContent)) + self.baseLineNo
        elseContent = self.content[ifBound[2]+1:]
        elseContent = elseContent[elseContent.find("else") + 4:]
        elseContentBaseLineNo = getLineNoByPos(self.content, self.content.index(elseContent)) + self.baseLineNo

        previousStmt = None
        for x in statement.scanString(ifContent):
            relativeLineNo = getLineNoByPos(self.content, x[1])
            state = parseStatement(ifContent[x[1]:x[2]], ifContentBaseLineNo + relativeLineNo, self)
            if state is not None:
                state.previous = previousStmt
                if previousStmt is not None:
                    previousStmt.next = state
                self.ifStatementList.append(state)
                previousStmt = state
        previousStmt = None
        for x in statement.scanString(elseContent):
            relativeLineNo = getLineNoByPos(self.content, x[1])
            state = parseStatement(elseContent[x[1]:x[2]], elseContentBaseLineNo + relativeLineNo)
            if state is not None:
                if previousStmt is not None:
                    previousStmt.next = state
                self.elseStatementList.append(state)
                previousStmt = state
        self.children = [self.ifStatementList, self.elseStatementList]

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
            self.initialization = parseStatement(self.content[leftParenPos+1:firstSemiPos]+';', self.baseLineNo, self)
        else:
            self.initialization = None
        if 'cond' in self.parseResultDict:
            self.loopCondition = parseStatement(self.content[firstSemiPos+1:secondSemiPos]+';', self.baseLineNo, self)
        else:
            self.loopCondition = None
        if 'post' in self.parseResultDict:
            self.postStatement = parseStatement(self.content[secondSemiPos+1:rightParenPos]+';', self.baseLineNo, self)
        else:
            self.postStatement = None
        forExclude = (Literal("for") + LParen + Optional((declaration | assignment)) + semicolon +
              Optional(expression) + semicolon + Optional((assignment | expression)) + RParen)
        forExclude = list(forExclude.scanString(self.content))[0]
        stateContent = self.content[forExclude[2]+1:]

        previousStmt = None
        for x in statement.scanString(stateContent):
            relativeLineNo = getLineNoByPos(self.content, x[1])
            state = parseStatement(stateContent[x[1]:x[2]], self.baseLineNo + relativeLineNo, self)
            if state is not None:
                state.previous = previousStmt
                if previousStmt is not None:
                    previousStmt.next = state
                self.statementList.append(state)
                previousStmt = state
        self.children = self.statementList


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
        self.loopCondition = Expression(whileContent[whileContent.find('(')+1 : whileContent.find(')')], self.baseLineNo, self)
        stateContent = self.content[whileExclude[2]+1:]

        previousStmt = None
        for x in statement.scanString(stateContent):
            relativeLineNo = getLineNoByPos(self.content, x[1])
            state = parseStatement(stateContent[x[1]:x[2]], self.baseLineNo + relativeLineNo, self)
            if state is not None:
                state.previous = previousStmt
                if previousStmt is not None:
                    previousStmt.next = state
                self.statementList.append(state)
                previousStmt = state
        self.children = self.statementList

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
        self.loopCondition = parseStatement(whileContent[whileContent.find('(')+1:whileContent.find(')')]+';',
                                            self.baseLineNo + getLineNoByPos(self.content, self.content.index("while")), self)
        stateContent = self.content[:whileExclude[1]]

        previousStmt = None
        for x in statement.scanString(stateContent):  # the content of each statement
            relativeLineNo = getLineNoByPos(self.content, x[1])
            state = parseStatement(self.content[x[1]:x[2]], relativeLineNo + self.baseLineNo, self)
            if state is not None:
                state.previous = previousStmt
                if previousStmt is not None:
                    previousStmt.next = state
                self.statementList.append(state)
                previousStmt = state
        self.children = self.statementList

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
    def __init__(self, content, baseLineNo, parent):
        super().__init__(content, baseLineNo, parent)
        self.childNode = None
        self.parseExpression()

    # todo
    def getRelatedVariable(self):
        pass

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
                Uop = UnaryOp(postfixTokens[index], postfixTokens[index-1], self.baseLineNo)
                postfixTokens[index-1] = Uop
                del postfixTokens[index]
                continue

            # Meet binary operator, pop
            elif postfixTokens[index] in binaryOperator:
                if index - 2 < 0:
                    fail("Failed to construct AST for expression!: Binary Operator")
                Bop = BinOp(postfixTokens[index], postfixTokens[index-2], postfixTokens[index-1], self.baseLineNo)
                postfixTokens[index-2] = Bop
                del postfixTokens[index]
                del postfixTokens[index-1]
                index -= 1
                continue

            # Meet ternary operator, pop
            elif postfixTokens[index] in ternaryOperator:
                if index - 3 < 0:
                    fail("Failed to construct AST for expression!: Binary Operator")
                Top = TernaryOp(postfixTokens[index], postfixTokens[index-3], postfixTokens[index-2], postfixTokens[index-1], self.baseLineNo)
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
                FunCall = FunctionCall(self.content, funcName, postfixTokens[index - arguNum:index], self.baseLineNo)
                postfixTokens[index-arguNum] = FunCall
                for i in range(0, arguNum):
                    del postfixTokens[index-i]
                index -= (arguNum - 1)
                continue

            elif postfixTokens[index] == '[':
                if index - 2 < 0:
                    fail("Failed to construct AST for expression!: Array")
                Arr = Array(postfixTokens[index - 2], postfixTokens[index - 1], self.baseLineNo)
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
            postfixTokens[0] = FunctionCall(None, funcName, arguList, self.baseLineNo)

        self.childNode = postfixTokens[0]


# todo content is wrong!?
class Array(ASTNode):
    def __init__(self, arr, index, baseLineNo):
        super().__init__('Array', baseLineNo, self)
        self.arr = arr
        self.index = index

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + outputNode(self.arr) + '[' + outputNode(self.index) + ']'


class FunctionCall(ASTNode):
    def __init__(self, content, funcName, arguList, baseLineNo):
        super().__init__(content, baseLineNo, self)
        self.funcName = funcName
        self.arguList = arguList
        # content = funcName + str(arguList).replace('[', '(').replace(']', ')')
        # super().__init__('content')

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + self.content


class UnaryOp(ASTNode):
    def __init__(self, op, opr, baseLineNo):
        super().__init__(op, baseLineNo, self)
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
    def __init__(self, op, left, right, baseLineNo):
        super().__init__(op, baseLineNo, self)
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
    def __init__(self, op, cond, yes, no, baseLineNo):
        super().__init__(op, baseLineNo, self)
        self.op = op
        self.cond = cond
        self.yes = yes
        self.no = no

    def output(self, indentLevel=0):
        indent = indentLevel * '\t'
        return indent + self.content


# def kernelAST(path, funcName):
#     P = Program(path)  # Build AST with P as the root node
#     for function in P.functionList:
#         if function.fnName == funcName:
#             return function
#     print('No such kernel function matched!')
#     return None


# can only find var defined in the func or parameter
def findTypeOfVariable(vName, fnRoot):
    for t, paraName in fnRoot.argList:
        if paraName == vName:
            return t
    # use bfs traversal because ...
    for stmt in bfsTraverseStmt(fnRoot):
        if type(stmt) == Declaration and stmt.varName == vName:
            return stmt.dateType
    return None


def variableIsRelatedToTID(vName, fnRoot, appearStmt):
    relatedVariable = ['threadIdx', 'blockIdx']
    for stmt in bfsTraverseStmt(fnRoot):
        if type(stmt) == Assignment:
            rightContent = stmt.rightSide.output()
            for v in relatedVariable:
                # todo not solid
                if v in rightContent:
                    leftNode = stmt.leftSide.childNode
                    # todo not solid
                    relatedVariable.append(leftNode if type(leftNode) == str else leftNode.arr)
                    break
        if stmt == appearStmt:
            if vName in relatedVariable:
                return True
    return False


# todo not enough!!
def isMainPath(fnRoot, stmt):
    parentPointer = stmt
    while parentPointer.parent != fnRoot:
        parentPointer = parentPointer.parent
        if type(parentPointer) == If:
            return False
        elif type(parentPointer) == For:
            return False
        elif type(parentPointer) == While:
            return False
        elif type(parentPointer) == DoWhile:
            return False
    return True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError or TypeError:
        pass
    return False


# todo miss functionCall
# should not pass expNode, but expNode.childNode!!!
def getExpVariable(opNode, resList):
    if type(opNode) == BinOp:
        leftSide = opNode.left
        rightSide = opNode.right
        if type(leftSide) == str and not is_number(leftSide):
            resList.append(leftSide)
        elif type(leftSide) != str:
            resList = getExpVariable(leftSide, resList)
        if type(rightSide) == str and not is_number(rightSide):
            resList.append(rightSide)
        elif type(rightSide) != str:
            resList = getExpVariable(rightSide, resList)
    elif type(opNode) == Array:
        resList.append(opNode.arr)
        idx = opNode.index
        if type(idx) == str and not is_number(idx):
            resList.append(idx)
        elif type(idx) != str:
            resList = getExpVariable(idx, resList)
    elif type(opNode) == UnaryOp:
        operand = opNode.operand
        if type(operand) == str and not is_number(operand):
            resList.append(operand)
        elif type(operand) != str:
            resList = getExpVariable(operand, resList)
    return resList


def replaceStmt(originStmt, newStmts):
    targetStmtList = originStmt.parent.children
    if type(targetStmtList[0]) == list:  # parent is IF
        if originStmt in targetStmtList[0]:
            targetStmtList = targetStmtList[0]
        elif originStmt in targetStmtList[1]:
            targetStmtList = targetStmtList[1]
        else:
            print("can't find the originStmt!!!")
            return None
    idx = targetStmtList.index(originStmt)
    if originStmt.previous is not None:
        originStmt.previous.next = newStmts[0]
        newStmts[0].previous = originStmt.previous.next
    if originStmt.next is not None:
        originStmt.next.previous = newStmts[-1]
        newStmts[-1].next = originStmt.next.previous
    for i, stmt in enumerate(newStmts):
        if i+1 < len(newStmts):
            newStmts[i].next = newStmts[i+1]
            newStmts[i + 1].previous = newStmts[i]
    targetStmtList.remove(originStmt)
    for stmt in reversed(newStmts):
        targetStmtList.insert(idx,stmt)


def insertAfterStmt(originStmt, newStmts):
    targetStmtList = originStmt.parent.children
    if type(targetStmtList[0]) == list:  # parent is IF
        if originStmt in targetStmtList[0]:
            targetStmtList = targetStmtList[0]
        elif originStmt in targetStmtList[1]:
            targetStmtList = targetStmtList[1]
        else:
            print("can't find the originStmt!!!")
            return None
    idx = targetStmtList.index(originStmt)
    originStmt.next = newStmts[0]
    newStmts[0].previous = originStmt
    if originStmt.next is not None:
        originStmt.next.previous = newStmts[-1]
        newStmts[-1].next = originStmt.next.previous
    for i, stmt in enumerate(newStmts):
        if i+1 < len(newStmts):
            newStmts[i].next = newStmts[i+1]
            newStmts[i + 1].previous = newStmts[i]
    for stmt in newStmts:
        targetStmtList.insert(idx+1, stmt)


def transToIf(originalStmt, conditionContent):
    wrappedStmt = None
    newIfContent = "if (%s){" % conditionContent + originalStmt.output() + "}"
    newIfObj = parseStatement(newIfContent, -1, originalStmt.parent)
    wrappedStmt = newIfObj.children[0][0]
    return newIfObj, wrappedStmt


def repair(fnContent, bugLineNo: list, bugType: int):
    func = FunctionDeclare(fnContent, startLineNo, None)
    ivCount = 0  # intermediate variable
    bugStmt = []
    # avoid return/break in advance
    returnFlag = False
    exitEarlyVariable = None
    for stmt in [x for x in preOrderTraverseStmt(func)]:
        if not returnFlag:
            if stmt.baseLineNo > max(bugLineNo):
                break
            if stmt.baseLineNo in bugLineNo:
                bugStmt.append(stmt)
            # todo: not solid
            if type(stmt) == Return and "return" in stmt.content:
                exitEarlyVariable = '__tmp%d_' % ivCount  # intermediate variable
                ivCount += 1
                initStmt = parseStatement("bool %s = false;" % exitEarlyVariable, -1, func)
                func.children[0].previous = initStmt
                initStmt.next = func.children[0]
                func.children.insert(0, initStmt)
                returnFlag = True
                newStmt = parseStatement("%s = true;" % exitEarlyVariable, -1, stmt.parent)
                replaceStmt(stmt, [newStmt])
        else:
            newIfStmt, wrappedStmt = transToIf(stmt, "! %s" % exitEarlyVariable)
            replaceStmt(stmt, [newIfStmt])
            if stmt.baseLineNo in bugLineNo:
                bugStmt.append(wrappedStmt)
            if stmt.baseLineNo == max(bugLineNo):
                pointer = newIfStmt
                while pointer.parent != func:
                    pointer = pointer.parent
                returnStmt = parseStatement("if(%s){ return; }"%exitEarlyVariable, -1, func)
                insertAfterStmt(pointer, [returnStmt])
                break

    # DR
    if bugType == 0:
        # DR in one line
        if len(bugLineNo) == 1:
            # must be assignment
            bugStmt = bugStmt[0]
            if type(bugStmt) == Assignment:
                originalContent = bugStmt.output()
                ivType = None
                exitEarlyVariable = '__tmp%d_' % ivCount  # intermediate variable
                ivCount += 1
                rightExp = bugStmt.rightSide
                expType = type(rightExp.childNode)
                # todo other type? or high dimensional array?
                if expType == Array:
                    arrType = findTypeOfVariable(rightExp.childNode.arr, func).strip()
                    if arrType[-1] == '*':
                        ivType = arrType[:-1]

                newAssignment1 = parseStatement(ivType+" "+exitEarlyVariable+" "+originalContent[originalContent.find('='):] + ';',
                                                -100, bugStmt.parent)
                newAssignment2 = parseStatement(originalContent[:originalContent.find('=')+1]+" "+exitEarlyVariable+';',
                                                -100, bugStmt.parent)
                barrierFn = parseStatement('__syncthreads();', -100, bugStmt.parent)
                barrierFn.previous = newAssignment1
                newAssignment2.previous = barrierFn
                newAssignment1.next = barrierFn
                barrierFn.next = newAssignment2
                if bugStmt.previous is not None:
                    bugStmt.previous.next = newAssignment1
                if bugStmt.next is not None:
                    bugStmt.next.previous = newAssignment2

                targetStmtList = bugStmt.parent.children
                if type(targetStmtList[0]) == list:  # if statement
                    targetStmtList = targetStmtList[0] if bugStmt in targetStmtList[0] else targetStmtList[1]
                targetPosition = targetStmtList.index(bugStmt)
                targetStmtList.remove(bugStmt)
                targetStmtList.insert(targetPosition, newAssignment2)
                targetStmtList.insert(targetPosition, barrierFn)
                targetStmtList.insert(targetPosition, newAssignment1)
        elif len(bugLineNo) == 2:
            bugStmt1 = getStmtByLineNo(func, bugLineNo[0])
            bugStmt2 = getStmtByLineNo(func, bugLineNo[1])
            isMain1 = isMainPath(func, bugStmt1)
            isMain2 = isMainPath(func, bugStmt2)
            if not isMain1 and not isMain2:
                if type(bugStmt1.parent) == If:
                    originalIfStmt = bugStmt1.parent
                    if bugStmt1 in originalIfStmt.children[0] and bugStmt2 in originalIfStmt.children[0]:
                        pass
                    elif bugStmt1 in originalIfStmt.children[0] and bugStmt2 in originalIfStmt.children[1]:
                        pass
                    elif bugStmt1 in originalIfStmt.children[1] and bugStmt2 in originalIfStmt.children[1]:
                        pass
    return func.output()


if __name__ == "__main__":
    ParserElement.enablePackrat()
    # fnName = "_copy_low_upp"
    fnName = "test"
    # sourcePath = "/home/instein/桌面/毕设/CUDA_Parser/test.cpp"
    sourcePath = "/home/instein/桌面/毕设/CUDA_Parser/tests/DR-1-exitEarlyInMain.cu"
    bugLineNo = [6, 7]  # may happen in two line [79, 80]
    # bugLineNo = [4, 5]
    bugType = 0  # 0:DR 1:BD 2:RB
    parser = Parser()
    with open(sourcePath) as source:
        content = source.read()
        frontPart, fnContent, endPart, startLineNo = parser.getFunctionContent(fnName, content)

    repairedFnContent = repair(fnContent, bugLineNo, bugType)
    print(repairedFnContent)
    targetPath = sourcePath[:sourcePath.rindex('.')] + '-fix.cu'
    with open(targetPath, 'w') as target:
        target.write(frontPart + repairedFnContent + endPart)
