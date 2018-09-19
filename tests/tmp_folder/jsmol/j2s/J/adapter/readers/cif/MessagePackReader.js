Clazz.declarePackage ("J.adapter.readers.cif");
Clazz.load (null, "J.adapter.readers.cif.MessagePackReader", ["java.lang.Boolean", "$.Float", "$.Long", "java.util.Hashtable", "JU.BC"], function () {
c$ = Clazz.decorateAsClass (function () {
this.doc = null;
this.isHomo = false;
Clazz.instantialize (this, arguments);
}, J.adapter.readers.cif, "MessagePackReader");
Clazz.makeConstructor (c$, 
function (binaryDoc, isHomogeneousArrays) {
this.isHomo = isHomogeneousArrays;
this.doc = binaryDoc;
}, "javajs.api.GenericBinaryDocument,~B");
Clazz.defineMethod (c$, "readMap", 
function () {
return this.getNext (null, 0);
});
Clazz.defineMethod (c$, "getNext", 
function (array, pt) {
var b = this.doc.readByte () & 0xFF;
var be0 = b & 0xE0;
if ((b & 128) == 0) {
if (array != null) {
(array)[pt] = b;
return null;
}return Integer.$valueOf (b);
}switch (be0) {
case 224:
b = JU.BC.intToSignedInt (b | 0xFFFFFF00);
if (array != null) {
(array)[pt] = b;
return null;
}return Integer.$valueOf (b);
case 160:
{
var s = this.doc.readString (b & 0x1F);
if (array != null) {
(array)[pt] = s;
return null;
}return s;
}case 128:
return ((b & 0xF0) == 128 ? this.getMap (b & 0x0F) : this.getArray (b & 0x0F));
case 192:
switch (b) {
case 192:
return null;
case 194:
return Boolean.FALSE;
case 195:
return Boolean.TRUE;
case 199:
{
var n = this.doc.readUInt8 ();
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (n)]);
}case 200:
{
var n = this.doc.readUnsignedShort ();
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (n)]);
}case 201:
{
var n = this.doc.readInt ();
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (n)]);
}case 212:
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (1)]);
case 213:
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (2)]);
case 214:
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (4)]);
case 215:
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (8)]);
case 216:
return  Clazz.newArray (-1, [Integer.$valueOf (this.doc.readUInt8 ()), this.doc.readBytes (16)]);
case 220:
return this.getArray (this.doc.readUnsignedShort ());
case 221:
return this.getArray (this.doc.readInt ());
case 222:
return this.getMap (this.doc.readUnsignedShort ());
case 223:
return this.getMap (this.doc.readInt ());
case 196:
return this.doc.readBytes (this.doc.readUInt8 ());
case 197:
return this.doc.readBytes (this.doc.readUnsignedShort ());
case 198:
return this.doc.readBytes (this.doc.readInt ());
}
if (array == null) {
switch (b) {
case 202:
return Float.$valueOf (this.doc.readFloat ());
case 203:
return Float.$valueOf (this.doc.readDouble ());
case 204:
return Integer.$valueOf (this.doc.readUInt8 ());
case 205:
return Integer.$valueOf (this.doc.readUnsignedShort ());
case 206:
return Integer.$valueOf (this.doc.readInt ());
case 207:
return Long.$valueOf (this.doc.readLong ());
case 208:
return Integer.$valueOf (this.doc.readByte ());
case 209:
return Integer.$valueOf (this.doc.readShort ());
case 210:
return Integer.$valueOf (this.doc.readInt ());
case 211:
return Long.$valueOf (this.doc.readLong ());
case 217:
return this.doc.readString (this.doc.readUInt8 ());
case 218:
return this.doc.readString (this.doc.readShort ());
case 219:
return this.doc.readString (this.doc.readInt ());
}
} else {
switch (b) {
case 202:
(array)[pt] = this.doc.readFloat ();
break;
case 203:
(array)[pt] = this.doc.readDouble ();
break;
case 204:
(array)[pt] = this.doc.readUInt8 ();
break;
case 205:
(array)[pt] = this.doc.readUnsignedShort ();
break;
case 206:
(array)[pt] = this.doc.readInt ();
break;
case 207:
(array)[pt] = this.doc.readLong ();
break;
case 208:
(array)[pt] = this.doc.readByte ();
break;
case 209:
(array)[pt] = this.doc.readShort ();
break;
case 210:
(array)[pt] = this.doc.readInt ();
break;
case 211:
(array)[pt] = this.doc.readLong ();
break;
case 217:
(array)[pt] = this.doc.readString (this.doc.readUInt8 ());
break;
case 218:
(array)[pt] = this.doc.readString (this.doc.readShort ());
break;
case 219:
(array)[pt] = this.doc.readString (this.doc.readInt ());
break;
}
}}
return null;
}, "~O,~N");
Clazz.defineMethod (c$, "getArray", 
 function (n) {
if (this.isHomo) {
if (n == 0) return null;
var v = this.getNext (null, 0);
if (Clazz.instanceOf (v, Integer)) {
var a =  Clazz.newIntArray (n, 0);
a[0] = (v).intValue ();
v = a;
} else if (Clazz.instanceOf (v, Float)) {
var a =  Clazz.newFloatArray (n, 0);
a[0] = (v).floatValue ();
v = a;
} else if (Clazz.instanceOf (v, String)) {
var a =  new Array (n);
a[0] = v;
v = a;
} else {
var o =  new Array (n);
o[0] = v;
for (var i = 1; i < n; i++) o[i] = this.getNext (null, 0);

return o;
}for (var i = 1; i < n; i++) this.getNext (v, i);

return v;
}var o =  new Array (n);
for (var i = 0; i < n; i++) o[i] = this.getNext (null, 0);

return o;
}, "~N");
Clazz.defineMethod (c$, "getMap", 
 function (n) {
var map =  new java.util.Hashtable ();
for (var i = 0; i < n; i++) {
var key = this.getNext (null, 0).toString ();
var value = this.getNext (null, 0);
if (value == null) {
} else {
map.put (key, value);
}}
return map;
}, "~N");
Clazz.defineStatics (c$,
"POSITIVEFIXINT_x80", 0x80,
"FIXMAP_xF0", 0x80,
"FIXSTR_xE0", 0xa0,
"NEGATIVEFIXINT_xE0", 0xe0,
"DEFINITE_xE0", 0xc0,
"NIL", 0xc0,
"FALSE", 0xc2,
"TRUE", 0xc3,
"BIN8", 0xc4,
"BIN16", 0xc5,
"BIN32", 0xc6,
"EXT8", 0xc7,
"EXT16", 0xc8,
"EXT32", 0xc9,
"FLOAT32", 0xca,
"FLOAT64", 0xcb,
"UINT8", 0xcc,
"UINT16", 0xcd,
"UINT32", 0xce,
"UINT64", 0xcf,
"INT8", 0xd0,
"INT16", 0xd1,
"INT32", 0xd2,
"INT64", 0xd3,
"FIXEXT1", 0xd4,
"FIXEXT2", 0xd5,
"FIXEXT4", 0xd6,
"FIXEXT8", 0xd7,
"FIXEXT16", 0xd8,
"STR8", 0xd9,
"STR16", 0xda,
"STR32", 0xdb,
"ARRAY16", 0xdc,
"ARRAY32", 0xdd,
"MAP16", 0xde,
"MAP32", 0xdf);
});
