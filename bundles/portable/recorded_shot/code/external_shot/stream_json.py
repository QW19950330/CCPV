"""Bounded standard-library JSON-object/array reader; one item at a time."""
import codecs,hashlib,json
def unique_pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON object key: '+k)
        out[k]=v
    return out
def reject_constant(value):raise ValueError('nonstandard JSON constant: '+value)
class Stream:
    def __init__(self,handle,chunk=1048576,max_item=8388608):
        self.handle=handle;self.chunk=chunk;self.max_item=max_item;self.buf='';self.pos=0;self.eof=False
        self.decode=codecs.getincrementaldecoder('utf8')('strict');self.decoder=json.JSONDecoder(object_pairs_hook=unique_pairs,parse_constant=reject_constant)
        self.hash=hashlib.sha256();self.bytes=0;self.max_live_chars=0
    def fill(self):
        if self.eof:return False
        self.buf=self.buf[self.pos:];self.pos=0
        b=self.handle.read(self.chunk);self.hash.update(b);self.bytes+=len(b)
        self.eof=not b;self.buf+=self.decode.decode(b,final=self.eof)
        self.max_live_chars=max(self.max_live_chars,len(self.buf))
        if len(self.buf)>self.max_item+self.chunk:raise ValueError('bounded JSON item limit exceeded')
        return bool(b)
    def peek(self):
        while True:
            while self.pos<len(self.buf) and self.buf[self.pos] in ' \t\r\n':self.pos+=1
            if self.pos<len(self.buf):return self.buf[self.pos]
            if self.eof:return ''
            self.fill()
    def take(self,c):
        if self.peek()!=c:raise ValueError('expected JSON delimiter '+repr(c))
        self.pos+=1
    def value(self):
        self.peek()
        while True:
            try:
                v,end=self.decoder.raw_decode(self.buf,self.pos)
                # Top-level values/items here are objects or strings, not undelimited numeric scalars.
                self.pos=end;return v
            except json.JSONDecodeError:
                if self.eof:raise
                if len(self.buf)-self.pos>=self.max_item:raise ValueError('oversized or malformed JSON item')
                self.fill()
    def records(self):
        self.take('{');seen=set()
        while self.peek()!='}':
            key=self.value()
            if not isinstance(key,str) or key in seen:raise ValueError('invalid/duplicate root key')
            seen.add(key);self.take(':')
            if key=='info':
                val=self.value()
                if not isinstance(val,dict):raise ValueError('info must be object')
                yield key,val
            elif key in ('images','annotations','categories'):
                self.take('[')
                while self.peek()!=']':
                    val=self.value()
                    if not isinstance(val,dict):raise ValueError('array item must be object')
                    yield key,val
                    if self.peek()==']':break
                    self.take(',')
                    if self.peek()==']':raise ValueError('trailing array comma')
                self.take(']');yield key+'__end',None
            else:raise ValueError('unplanned top-level JSON key '+key)
            if self.peek()=='}':break
            self.take(',')
            if self.peek()=='}':raise ValueError('trailing root comma')
        self.take('}')
        if self.peek()!='':raise ValueError('trailing data after JSON document')
        if seen!={'info','images','annotations','categories'}:raise ValueError('missing required COCO top-level key')
