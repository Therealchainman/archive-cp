
class Solver:
    def __init__(self, hexa):
        self.version = 0
        self.i = 0
        self.result = 0
    
    def data_loader(self):
        with open("inputs/input.txt", "r") as f:
            return f.read()

    def conv_hexa_binary(self, hexa):
        hexBin = {'0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100', '5': '0101', '6': '0110', 
        '7': '0111', '8': '1000', '9': '1001', 'A': '1010', 'B': '1011', 'C': '1100', 'D': '1101', 'E': '1110', 'F': '1111'}
        binary = ""
        for c in hexa:
            binary += hexBin[c]
        return binary

    def is_literal(self, hexa):
        return hexa == 4

    def is_operator(self, hexa):
        pass

    def get_version(self, hexa):
        return int(hexa, 2)

    def parse_literal(self, hexa):
        binary = ""
        while True:
            A = hexa[self.i:self.i+5]
            self.i += 5
            binary += A[1:]
            if A[0] == '0':
                break
        return int(binary, 2)

    def parse_operator(self, typ, hexa):
        if typ == 0:
            total = 0
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1   
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                starting_bit = self.i
                while self.i-starting_bit<num_bits:
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total += lval
                    else:
                        op = self.parse_operator(pid,hexa)
                        total += op
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(num_subpackets):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total += lval
                    else:
                        op = self.parse_operator(pid,hexa)
                        total += op
            return total
        elif typ==1:
            total = 1
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1   
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                starting_bit = self.i
                while self.i-starting_bit<num_bits:
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total*=lval
                    else:
                        op = self.parse_operator(pid,hexa)
                        total*=op
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(num_subpackets):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total*=lval
                    else:
                        op = self.parse_operator(pid,hexa)
                        total*=op
            return total
        elif typ==2:
            total = 1000000
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1   
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                starting_bit = self.i
                while self.i-starting_bit<num_bits:
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total = min(lval,total)
                    else:
                        op = self.parse_operator(pid,hexa)
                        total = min(op, total)
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(num_subpackets):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total = min(total,lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        total = min(total,op)
            return total
        elif typ==3:
            total = -1000000
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1   
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                starting_bit = self.i
                while self.i-starting_bit<num_bits:
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total = max(total,lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        total = max(total,op)
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(num_subpackets):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        total = max(total,lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        total = max(total,op)
            return total
        elif typ==5:
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1  
            data = [] 
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                for _ in range(2):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        data.append(lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        data.append(op)
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(2):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        data.append(lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        data.append(op)
            return 1 if data[0]>data[1] else 0
        elif typ==6:
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1  
            data = [] 
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                for _ in range(2):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        data.append(lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        data.append(op)
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(2):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        data.append(lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        data.append(op)
            return 1 if data[0]<data[1] else 0
        elif typ==7:
            len_id = self.get_length_id(hexa[self.i])
            self.i+=1  
            data = [] 
            if len_id==0:
                num_bits = self.get_len_subpackets(hexa[self.i:self.i+15])
                self.i += 15
                for _ in range(2):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        data.append(lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        data.append(op)
            else:
                num_subpackets = self.get_num_subpackets(hexa[self.i:self.i+11])
                self.i +=11
                for _ in range(2):
                    v = self.get_version(hexa[self.i:self.i+3])
                    self.version += v
                    self.i += 3
                    pid = self.get_packet_type(hexa[self.i:self.i+3])
                    self.i += 3
                    if self.is_literal(pid):
                        lval = self.parse_literal(hexa)
                        data.append(lval)
                    else:
                        op = self.parse_operator(pid,hexa)
                        data.append(op)
            return 1 if data[0]==data[1] else 0
    def get_length_id(self, hexa):
        return int(hexa,2)

    def get_num_subpackets(self, hexa):
        return int(hexa,2)

    def get_len_subpackets(self, hexa):
        return int(hexa,2)
    def get_packet_type(self, hexa):
        return int(hexa, 2)
    def run(self):
        binary_data = self.conv_hexa_binary(self.data_loader())
        print(binary_data)
        while self.i<len(binary_data):
            v = self.get_version(binary_data[self.i:self.i+3])
            self.version += v
            self.i += 3
            pid = self.get_packet_type(binary_data[self.i:self.i+3])
            self.i += 3
            if self.is_literal(pid):
                lval = self.parse_literal(binary_data)
            else:
                op = self.parse_operator(pid, binary_data)
                self.result += op
            return self.result



        return self.version
    
if __name__ == '__main__':
    s = Solver(None)
    print(s.run())


"""
0 = 0000
1 = 0001
2 = 0010
3 = 0011
4 = 0100
5 = 0101
6 = 0110
7 = 0111
8 = 1000
9 = 1001
A = 1010
B = 1011
C = 1100
D = 1101
E = 1110
F = 1111
38006F45291200
3 -> 0011
8 -> 1000

VVVTTT , 4 then it's literal value

00111000000000000110111101000101001010010001001000000000
VVVTTTILLLLLLLLLLLLLLLAAAAAAAAAAABBBBBBBBBBBBBBBB
"""