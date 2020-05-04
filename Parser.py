from Utils import *
import re

class Parser:
    def __init__(self, sourcePath=None, fnName=None):
        self.sourcePath = sourcePath
        self.fnName = fnName
        self.dataType = []

    def getFunction(self, source, fnName):
        with open(source) as sourceFile:
            for line in sourceFile:
                pass


    def basicCheck(self, token):
        varPtrn = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]")  # variables
        headerPtrn = re.compile(r"\w[a-zA-Z]+[.]h")  # header files
        digitPtrn = re.compile(r'\d')
        floatPtrn = re.compile(r'\d+[.]\d+')

        if token in keywords():
            print(token + " KEYWORD")
        elif token in operators().keys():
            print(token + " ", operators()[token])
        elif token in delimiters():
            description = delimiters()[token]
            if description == 'TAB' or description == 'NEWLINE':
                print(description)
            else:
                print(token + " ", description)
        elif re.search(headerPtrn, token):
            print(token + " HEADER")
        elif re.match(varPtrn, token) or "'" in token or '"' in token:
            print(token + ' IDENTIFIER')
        elif re.match(digitPtrn, token):
            if re.match(floatPtrn, token):
                print(token + ' FLOAT')
            else:
                print(token + ' INT')
        return True

    def delimiterCorrection(self, line):
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
        for token in tokens:
            if self.isWhiteSpace(token):
                tokens.remove(token)
            # elif ' ' in token:
            #     tokens.remove(token)
            #     token = token.split(' ')
            #     for d in token:
            #         tokens.append(d)
        return tokens

    def isWhiteSpace(self, word):
        ptrn = [" ", "\t", "\n", ""]
        for item in ptrn:
            if word == item:
                return True
        return False

    def tokenize(self, content):
        lines = content.split("\n")
        count = 0
        for line in lines:
            count = count + 1
            tokens = self.delimiterCorrection(line)
            print("\n#LINE ", count)
            print("Tokens: ", tokens)
            for token in tokens:
                self.basicCheck(token)
        return True

if __name__ == '__main__':
    parser = Parser()
    with open('D:/DATA/Python_ws/CUDA_Parser/test.cpp') as source:
        content = source.read()
        parser.tokenize(content)
