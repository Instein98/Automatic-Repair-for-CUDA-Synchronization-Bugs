from Grammar import *

def formatAST(unformattedAST):
    tempList = list(str(unformattedAST[0]))
    braceCount = -1
    for index, char in enumerate(tempList):
        if char == '\n':
            tempList.insert(index+1, (braceCount * '    ') if braceCount > 0 else '')
        elif char == '{':
            braceCount += 1
        elif char == '}':
            braceCount -= 1
    return "".join(tempList)  


def fail():
    print("!!!!!!!PARSE FAILED!!!!!!!!")


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
            self.expressionList.append(Expression(self.content[x[1]:x[2]]))
        if not self.expressionList:
            fail()


class Expression(ASTNode):
    def __init__(self, content):
        super().__init__(content)
        # self.parseExpression()

    def parseExpression(self):
        arithExpParseResult = arithmeticExp.parseString(self.content)
        print(" ")

    def parse_term(_term):
        pass

class UnaryOp(ASTNode):
    pass


class BinOp(ASTNode):
    def __init__(self, content, op, left, right):
        super().__init__(content)
        self.op = op
        self.left = left
        self.right = right


if __name__ == "__main__":
    dir = "D:/DATA/Python_ws/CUDA_Parser/test/stage_4/valid/"
    def test(path):
        P = Program(dir + path)
        print(P.functionList[0].statementList[0].content)

    # stage 4
    test("and_false.c")
    test("and_true.c")
    test("eq_false.c")
    test("eq_true.c")
    test("ge_false.c")
    test("ge_true.c")
    test("gt_false.c")
    test("gt_true.c")
    test("le_false.c")
    test("le_true.c")
    test("lt_false.c")
    test("lt_true.c")
    test("ne_false.c")
    test("ne_true.c")
    test("or_false.c")
    test("or_true.c")
    test("precedence.c")
    test("precedence_2.c")
    test("precedence_3.c")
    test("precedence_4.c")



