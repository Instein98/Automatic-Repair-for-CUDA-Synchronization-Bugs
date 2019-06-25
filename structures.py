
from grammar import *

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


class UnaryOp(ASTNode):
    pass


class BinOp(ASTNode):
    pass


def test():
    return 2 - - 3

# class Expression(ASTNode):
#     def assignFields(self):
#         self.num = self.tokens
#         del self.tokens


# class DataType(ASTNode):
#     def assignFields(self):
#         self.dataType = self.tokens
#         del self.tokens


# class Identifier(ASTNode):
#     def assignFields(self):
#         self.identifier = self.tokens
#         del self.tokens


if __name__ == "__main__":
    Path = "D:/DATA/Python_ws/CUDA_Parser/test/stage_3/valid/add.c"
    # Path = "D:/DATA/Python_ws/CUDA_Parser/test/stage_1/invalid/no_space.c"
    P = Program(Path)
    print(P.functionList[0].statementList[0].content)