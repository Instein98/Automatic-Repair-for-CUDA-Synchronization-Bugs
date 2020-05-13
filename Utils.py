import re

def keywords():
    keywords = [
        "auto",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "word",
        "class",
        "const",
        "continue",
        "delete",
        "do",
        "double",
        "else",
        "enum",
        "false",
        "float",
        "for",
        "goto",
        "if",
        "#include",
        "int",
        "long",
        "namespace",
        "not",
        "or",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "true",
        "try",
        "unsigned",
        "void",
        "while",
        "__global__",
        "template",
        "typename",
    ]
    return keywords

def operators():
    operators = {
        "+": "PLUS",
        "-": "MINUS",
        "*": "MUL",
        "/": "DIV",
        "%": "MOD",
        "+=": "PLUSEQ",
        "-=": "MINUSEQ",
        "*=": "MULEQ",
        "/=": "DIVEQ",
        "++": "INC",
        "--": "DEC",
        "|": "OR",
        "&&": "AND",
    }
    return operators

def delimiters():
    delimiters = {
        "\t": "TAB",
        "\n": "NEWLINE",
        "(": "LPAR",
        ")": "RPAR",
        "[": "LBRACE",
        "]": "RBRACE",
        "{": "LCBRACE",
        "}": "RCBRACE",
        "=": "ASSIGN",
        ":": "COLON",
        ",": "COMMA",
        ";": "SEMICOL",
        "<<": "OUT",
        ">>": "IN",
    }
    return delimiters


def getFunction(content, fnName):
    """
    template<typename Real>
    __device__
    static Real _sum_reduce(Real buffer[]) {
    :param content:
    :param fnName:
    :return:
    """
    pattern = re.compile(r"[\s\S]*?%s[\s]*\(.*?\)[\s]*\{([\s\S]*?)\}[\s\S]*?" % fnName)
    m = pattern.match(content)
    return m.group(1)


def getLineNoByPos(content, position):
    lineNo = 0
    for i, a in enumerate(content):
        if a == '\n':
            lineNo += 1
        if i == position:
            return lineNo
    return None


def getStmtByLineNo(node, lineNo):
    if node.baseLineNo == lineNo:
        return node
    if node.children is not None and len(node.children) != 0:  # mode.children must be list
        for child in node.children:
            if type(child) != list:
                res = getStmtByLineNo(child, lineNo)
                if res is not None:
                    return res
            else:
                for subchild in child:
                    res = getStmtByLineNo(subchild, lineNo)
                    if res is not None:
                        return res
    return None


# pre-order traversal
def preOrderTraverseStmt(node):
    yield node
    if node.children is not None and len(node.children) != 0:  # mode.children must be list
        for child in node.children:
            if type(child) != list:
                for res in preOrderTraverseStmt(child):
                    yield res
            else:
                for subchild in child:
                    for res in preOrderTraverseStmt(subchild):
                        yield res


# bfs traversal
def bfsTraverseStmt(node):
    queue = [node]
    while len(queue) > 0:
        currentNode = queue[0]
        yield currentNode
        queue.pop(0)
        # print(queue)
        if currentNode.children is not None and len(currentNode.children) != 0:  # mode.children must be list
            for child in currentNode.children:
                if type(child) != list:
                    queue.append(child)
                else:
                    queue.extend(child)
