from Grammar import *
from abc import ABC, abstractmethod


localVar = []
globalVar = []
unaryOperator = ['unary -', 'unary ~', 'unary !']
binaryOperator = ['+', '-', '*', '/', '>', '>=', '<', '<=', '==', '!=', '&&', '||']
ternaryOperator = ['?']

def fail(str = "!!!!!!! PARSE FAILED !!!!!!!!"):
    print(str)

def parseStatement(content):
    parseResult = statement.parseString(content)
    parseResultDict = parseResult.asDict()
    if 'If' in parseResultDict:
        return If(content, parseResult)
    if 'Return' in parseResultDict:
        return Return(content, parseResult)
    if 'Declaration' in parseResultDict:
        return Declaration(content, parseResult)
    if 'Assignment' in parseResultDict:
        return Assignment(content, parseResult)
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
        f = open(self.path)
        self.content = f.read()
        f.close()
        self.parseFunctions()

    def parseFunctions(self):
        parseResult = function.scanString(self.content)
        for x in parseResult:
            self.functionList.append(FunctionDeclare(self.content[x[1]:x[2]]))
        if not self.functionList:
            fail()


class FunctionDeclare(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.parseFunction()

    def parseFunction(self):
        parseResult = function.parseString(self.content)
        self.returnType = parseResult['returnType']
        self.fnName = parseResult['fnName']
        self.statementList = []
        scanResult = statement.scanString(self.content)
        for x in scanResult:  # the content of each statement
            state = parseStatement(self.content[x[1]:x[2]])
            if state != None:
                self.statementList.append(state)
        if not self.statementList:
            fail()




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
        self.leftSide = self.parseResultDict['left']
        self.rightSide = Expression(expToString(self.parseResultDict['right']))


class Return(Statement):
    def parse(self):
        self.returnExp = Expression(expToString(self.parseResultDict['retExp']))


class Declaration(Statement):
    def parse(self):
        self.dataType = self.parseResultDict['dataType']
        self.varName = self.parseResultDict['varName']
        if 'initialValue' in self.parseResultDict:
            self.initExp = Expression(expToString(self.parseResultDict['initialValue']))

class If(Statement):
    def parse(self):
        self.ifStatementList = []
        self.elseStatementList = []

        ifBound = list(ifPart.scanString(self.content))[0]
        ifContent = self.content[ifBound[1]:ifBound[2]]
        self.condExp = Expression(ifContent[ifContent.find('(')+1:ifContent.find(')')])
        ifExclude = list((Literal("if") + LParen + expression('condExp') + RParen).scanString(ifContent))[0]
        ifContent = ifContent[ifExclude[2]+1:]

        elseContent = self.content[ifBound[2]+1:]
        for x in statement.scanString(ifContent):
            state = parseStatement(ifContent[x[1]:x[2]])
            if state != None:
                self.ifStatementList.append(state)
        for x in statement.scanString(elseContent):
            state = parseStatement(elseContent[x[1]:x[2]])
            if state != None:
                self.elseStatementList.append(state)


class Expression(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.operator = None
        self.parseExpression()

    def parseExpression(self):
        postfixTokens = parseExp(self.content)
        print(self.content," -> ",postfixTokens)
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

            # Meet identifiers, just push
            else:
                index += 1
                if index > len(postfixTokens) - 1:
                    fail("Failed to construct AST for expression!: Index out of range")
        self.operator = postfixTokens[0]



class UnaryOp(ASTNode):
    def __init__(self, op, opr):
        self.operator = op
        self.operand = opr


class BinOp(ASTNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


class TernaryOp(ASTNode):
    """conditional expression: a ? b : c"""
    def __init__(self, op, cond, yes, no):
        self.op = op
        self.cond = cond
        self.yes = yes
        self.no = no


if __name__ == "__main__":
    stage = 6
    dir = "D:/DATA/Python_ws/CUDA_Parser/test/stage_%d/valid/" % stage
    def test(path):
        P = Program(dir + path)
        print('Succeed')

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





