from Utils import *
import re


class Token:
    def __init__(self, literal, lineNum, tokenType):  # pos like (1, 2)
        self.literal = literal
        self.lineNum = lineNum
        self.tokenType = tokenType

class Parser:
    def __init__(self, sourcePath=None, fnName=None, content=None):
        self.sourcePath = sourcePath
        self.fnName = fnName
        self.dataType = []
        self.content = content

    def basicCheck(self, token):
        varPtrn = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]")  # variables
        headerPtrn = re.compile(r"\w[a-zA-Z]+[.]h")  # header files
        digitPtrn = re.compile(r'\d')
        floatPtrn = re.compile(r'\d+[.]\d+')

        if token in keywords():
            # print(token + " KEYWORD")
            return "KEYWORD"
        elif token in operators().keys():
            # print(token + " ", operators()[token])
            return operators()[token]
        elif token in delimiters():
            description = delimiters()[token]
            if description == 'TAB' or description == 'NEWLINE':
                # print(description)
                return description
            else:
                # print(token + " ", description)
                return description
        elif re.search(headerPtrn, token):
            # print(token + " HEADER")
            return "HEADER"
        elif re.match(varPtrn, token) or "'" in token or '"' in token:
            # print(token + ' IDENTIFIER')
            return "IDENTIFIER"
        elif re.match(digitPtrn, token):
            if re.match(floatPtrn, token):
                # print(token + ' FLOAT')
                return "FLOAT"
            else:
                # print(token + ' INT')
                return "INT"
        return True

    def delimiterCorrection(self, line, lineNum):
        tokens = line.strip().split(" ")
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '':
                tokens.remove(token)
                continue
            for delimiter in delimiters().keys():
                if token != delimiter and delimiter in token:
                    pos = token.find(delimiter)
                    firstPart = token[:pos]  #if token[:pos] != "" else None
                    secondPart = token[pos+1:]
                    tokens.remove(token)
                    tokens.insert(i, secondPart)
                    tokens.insert(i, delimiter)
                    tokens.insert(i, firstPart)
                    i -= 1
                    break
            i += 1
        while i < len(tokens):
            if self.isWhiteSpace(tokens[i]):
                tokens.remove(tokens[i])
                i -= 1
            i += 1

            # elif ' ' in token:
            #     tokens.remove(token)
            #     token = token.split(' ')
            #     for d in token:
            #         tokens.append(d)
        for i in range(len(tokens)):
            token = tokens[i]
            tokens[i] = Token(token, lineNum, self.basicCheck(token))
        return tokens

    def isWhiteSpace(self, word):
        ptrn = [" ", "\t", "\n", ""]
        for item in ptrn:
            if word == item:
                return True
        return False

    def tokenize(self, content):
        res = []
        lines = content.split("\n")
        count = 0
        for line in lines:
            count = count + 1
            tokens = self.delimiterCorrection(line, count)
            # print("\n#LINE ", count)
            # print("Tokens: ", [ t.literal for t in tokens])
            res.extend(tokens)
            for token in tokens:
                res.append(token)
                # self.basicCheck(token)
        return res

    def getFunctionContent(self, functionName, content):
        fnStartLineNum = None
        tokenList = self.tokenize(content)
        for i, token in enumerate(tokenList):
            if token.literal == functionName and tokenList[i+1].literal == '(':
                j = i+2
                while tokenList[j].literal != '{' and tokenList[j].literal != ';':
                    j += 1
                if tokenList[j].literal == ';':
                    continue
                else:
                    if "__" in tokenList[i-2].literal:
                        fnStartLineNum = tokenList[i-2].lineNum
                        break
                    elif "__" in tokenList[i-3].literal:
                        fnStartLineNum = tokenList[i - 3].lineNum
                        break
                    else:
                        continue
        if fnStartLineNum is None:
            print("Function %s not found!!" % functionName)
            return None
        i = j
        stack = []
        stack.append('{')
        while len(stack) != 0:
            i += 1
            if tokenList[i].literal == '{':
                stack.append('{')
            if tokenList[i].literal == '}':
                stack.pop(0)
            else:
                continue
        fnEndLineNum = tokenList[i].lineNum

        fnPart = ''
        frontPart = ''
        endPart = ''
        lines = content.split("\n")
        for i, line in enumerate(lines):
            lineNum = i+1
            if lineNum < fnStartLineNum:
                frontPart += line + '\n'
            elif lineNum == fnStartLineNum:
                tmp = line.index('__')
                frontPart += line[:tmp]
                fnPart += line[tmp:] + '\n'
            elif fnStartLineNum < lineNum < fnEndLineNum:
                fnPart += line + '\n'
            elif lineNum == fnEndLineNum:
                tmp = line.rindex('}')
                fnPart += line[:tmp+1]
                endPart += line[tmp+1:] + '\n'
            elif lineNum > fnEndLineNum:
                endPart += line + '\n'
        return frontPart, fnPart, endPart, fnStartLineNum



if __name__ == '__main__':
    parser = Parser()
    with open('D:/DATA/Python_ws/CUDA_Parser/test.cpp') as source:
        content = source.read()
        fnContent = parser.getFunctionContent("_sum_reduce", content)
        # fnContent = getFunction(content, "_sum_reduce")
        print(fnContent)
