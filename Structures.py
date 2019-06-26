from Grammar import *

unaryOperator = ['unary -', 'unary ~', 'unary !']
binaryOperator = ['+', '-', '*', '/', '>', '>=', '<', '<=', '==', '!=', '&&', '||']


# def formatAST(unformattedAST):
#     tempList = list(str(unformattedAST[0]))
#     braceCount = -1
#     for index, char in enumerate(tempList):
#         if char == '\n':
#             tempList.insert(index+1, (braceCount * '    ') if braceCount > 0 else '')
#         elif char == '{':
#             braceCount += 1
#         elif char == '}':
#             braceCount -= 1
#     return "".join(tempList)


def fail(str = "!!!!!!!PARSE FAILED!!!!!!!!"):
    print(str)


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
        for x in scanResult:
            self.statementList.append(Statement(self.content[x[1]:x[2]]))
        if not self.statementList:
            fail()


class Statement(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.expressionList = []
        self.parseStatement()

    def parseStatement(self):
        scanResult = expression.scanString(self.content)
        for x in scanResult:
            if self.content[x[1]:x[2]] == "return ":
                continue
            elif self.content[x[1]:x[1]+7] == "return ":
                self.expressionList.append(Expression(self.content[x[1]+7:x[2]]))
                continue
            self.expressionList.append(Expression(self.content[x[1]:x[2]]))
        if not self.expressionList:
            fail()


class Expression(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        self.operator = None
        self.parseExpression()

    def parseExpression(self):
        postfixTokens = parseExp(self.content)
        print(self.content," -> ",postfixTokens)
        self.constructNode(postfixTokens)

    def constructNode(self, postfixTokens):
        index = 0
        while len(postfixTokens) > 1:
            if postfixTokens[index] in unaryOperator:
                if index - 1 < 0:
                    fail("Failed to construct AST for expression!: Unary Operator")
                Uop = UnaryOp(postfixTokens[index], postfixTokens[index-1])
                postfixTokens[index-1] = Uop
                del postfixTokens[index]
                continue
            elif postfixTokens[index] in binaryOperator:
                if index - 2 < 0:
                    fail("Failed to construct AST for expression!: Binary Operator")
                Bop = BinOp(postfixTokens[index], postfixTokens[index-2], postfixTokens[index-1])
                postfixTokens[index-2] = Bop
                del postfixTokens[index]
                del postfixTokens[index-1]
                index -= 1
                continue
            else:
                index += 1
                if index > len(postfixTokens) - 1:
                    fail("Failed to construct AST for expression!: Index out of range")
        self.operator = postfixTokens[0]



class UnaryOp(ASTNode):
    def __init__(self, op, opr):
        # content = str(op) + str(opr)
        # super().__init__(content)
        self.operator = op
        self.operand = opr


class BinOp(ASTNode):
    def __init__(self, op, left, right):
        # content = str(left) + str(op) + str(right)
        # super().__init__(content)
        self.op = op
        self.left = left
        self.right = right


if __name__ == "__main__":
    stage = 4
    dir = "D:/DATA/Python_ws/CUDA_Parser/test/stage_%d/valid/" % stage
    def test(path):
        P = Program(dir + path)
        print(P.functionList[0].statementList[0].content)

    # stage 3
    # test("add.c")

    # # stage 4
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
    test("ne_true.c")
    # test("or_false.c")
    # test("or_true.c")
    # test("precedence.c")
    # test("precedence_2.c")
    # test("precedence_3.c")
    # test("precedence_4.c")



