from Grammar import *
from abc import ABC, abstractmethod


unaryOperator = ['preUnary --', 'preUnary ++','preUnary ~', 'preUnary !', 'preUnary -', 'preUnary &',
                 'preUnary *', 'postUnary --', 'postUnary ++']
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


class ASTNode(object):
    def __init__(self, content):
        self.content = content  # origin text of the node

    def __str__(self):
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


class Statement(ASTNode, ABC):
    def __init__(self, content, parseResult):
        super().__init__(content)
        self.parseResult = parseResult
        self.parseResultDict = parseResult.asDict()
        self.parse()

    @abstractmethod
    def parse(self):
        pass


class Assignment(Statement):
    def parse(self):
        self.leftSide = Expression(self.content[:self.content.find('=')])
        self.rightSide = Expression(self.content[self.content.find('=')+1:].replace(';',''))


class Return(Statement):
    def parse(self):
        self.returnExp = Expression(self.content[self.content.find('return')+6:].replace(';',''))


class Declaration(Statement):
    def parse(self):
        self.declareList = None
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


class For(Statement):
    def parse(self):
        self.statementList = []
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


class Single(Statement):
    """break/continue etc."""
    def parse(self):
        pass


class Expression(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.childNode = None
        self.parseExpression()

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
                FunCall = FunctionCall(funcName, postfixTokens[index - arguNum:index])
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
            postfixTokens[0] = FunctionCall(funcName, arguList)

        self.childNode = postfixTokens[0]


class Array(ASTNode):
    def __init__(self, arr, index):
        super().__init__('Array')
        self.arr = arr
        self.index = index



class FunctionCall(ASTNode):
    def __init__(self, funcName, arguList):
        super().__init__(funcName)
        self.funcName = funcName
        self.arguList = arguList
        # content = funcName + str(arguList).replace('[', '(').replace(']', ')')
        # super().__init__('content')


class UnaryOp(ASTNode):
    def __init__(self, op, opr):
        super().__init__(op)
        self.operator = op
        self.operand = opr


class BinOp(ASTNode):
    def __init__(self, op, left, right):
        super().__init__(op)
        self.op = op
        self.left = left
        self.right = right


class TernaryOp(ASTNode):
    """conditional expression: a ? b : c"""
    def __init__(self, op, cond, yes, no):
        super().__init__(op)
        self.op = op
        self.cond = cond
        self.yes = yes
        self.no = no


def kernelAST(path, funcName):
    P = Program(path)  # Build AST with P as the root node
    for function in P.functionList:
        if function.fnName == funcName:
            return function
    print('No such kernel function matched!')
    return None




if __name__ == "__main__":
    # stage = 9
    # dir = "D:/DATA/Python_ws/CUDA_Parser/test/stage_%d/valid/" % stage
    # def test(path):
    #     P = Program(dir + path)
    #     print('Succeed')

    # stage 3
    # test("add.c")

    # stage 4
    # test("and_false.c")
    # test("and_true.c")
    # test("eq_false.c")
    # test("eq_true.c")
    # test("ge_false.c")
    # test("ge_true.c")
    # test("gt_false.c")
    # test("gt_true.c")
    # test("le_false.c")
    # test("le_true.c")
    # test("lt_false.c")
    # test("lt_true.c")
    # test("ne_false.c")
    # test("ne_true.c")
    # test("or_false.c")
    # test("or_true.c")
    # test("precedence.c")
    # test("precedence_2.c")
    # test("precedence_3.c")
    # test("precedence_4.c")

    # #stage 5
    # test("assign.c")

    #stage 6
    # test("statement/else.c")
    # test("statement/if_nested.c")
    # test("statement/if_nested_3.c")
    # test("statement/if_nested_5.c")
    # test("statement/if_taken.c")
    # test("statement/multiple_if.c")
    # test("expression/assign_ternary.c")
    # test("expression/multiple_ternary.c")

    # stage 8
    # test("break.c")
    # test("continue.c")
    # test("continue_empty_post.c")
    # test("do_while.c")
    # test("empty_expression.c")
    # test("for.c")
    # test("for_decl.c")
    # test("for_empty.c")
    # test("for_nested_scope.c")
    # test("for_variable_shadow.c")
    # test("nested_break.c")
    # test("nested_while.c")
    # test("return_in_while.c")

    # stage 9
    # test('expression_args.c')
    # test('fib.c')
    # test('forward_decl.c')
    # test('forward_decl_args.c')
    # test('forward_decl_multi_arg.c')
    # test('fun_in_expr.c')
    # test('hello_world.c')
    # test('multi_arg.c')
    # test('mutual_recursion.c')
    # test('no_arg.c')
    # test('precedence.c')
    # test('variable_as_arg.c')

    path = 'D:/DATA/Python_ws/CUDA_Parser/test.cpp'
    funcName = '_sum_reduce'
    functionAST = kernelAST(path, funcName)
    print("finish")

