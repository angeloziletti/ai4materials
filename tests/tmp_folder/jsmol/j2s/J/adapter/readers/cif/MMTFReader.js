Clazz.declarePackage ("J.adapter.readers.cif");
Clazz.load (["J.adapter.readers.cif.MMCifReader"], "J.adapter.readers.cif.MMTFReader", ["java.util.Hashtable", "JU.BC", "$.BS", "$.Lst", "$.M4", "$.PT", "$.SB", "J.adapter.readers.cif.MessagePackReader", "J.adapter.smarter.Atom", "$.Bond", "$.Structure", "JU.Logger"], function () {
c$ = Clazz.decorateAsClass (function () {
this.map = null;
this.fileAtomCount = 0;
this.opCount = 0;
this.groupModels = null;
this.labelAsymList = null;
this.atomMap = null;
Clazz.instantialize (this, arguments);
}, J.adapter.readers.cif, "MMTFReader", J.adapter.readers.cif.MMCifReader);
Clazz.overrideMethod (c$, "addHeader", 
function () {
});
Clazz.overrideMethod (c$, "setup", 
function (fullPath, htParams, reader) {
this.isBinary = true;
this.isMMCIF = true;
this.setupASCR (fullPath, htParams, reader);
}, "~S,java.util.Map,~O");
Clazz.overrideMethod (c$, "processBinaryDocument", 
function () {
var doDoubleBonds = (!this.isCourseGrained && !this.checkFilterKey ("NODOUBLE"));
this.applySymmetryToBonds = true;
this.map = ( new J.adapter.readers.cif.MessagePackReader (this.binaryDoc, true)).readMap ();
JU.Logger.info ("MMTF version " + this.map.get ("mmtfVersion"));
JU.Logger.info ("MMTF Producer " + this.map.get ("mmtfProducer"));
this.appendLoadNote (this.map.get ("title"));
var id = this.map.get ("structureId");
this.fileAtomCount = (this.map.get ("numAtoms")).intValue ();
var nBonds = (this.map.get ("numBonds")).intValue ();
JU.Logger.info ("id atoms bonds " + id + " " + this.fileAtomCount + " " + nBonds);
this.getAtoms (doDoubleBonds);
if (!this.isCourseGrained) {
this.getBonds (doDoubleBonds);
this.getStructure (this.map.get ("secStructList"));
}this.setSymmetry ();
this.getBioAssembly ();
this.setModelPDB (true);
});
Clazz.defineMethod (c$, "rldecode32", 
 function (b, n) {
if (b == null) return null;
var ret =  Clazz.newIntArray (n, 0);
for (var i = 0, pt = -1; i < n; ) {
var val = JU.BC.bytesToInt (b, (++pt) << 2, true);
for (var j = JU.BC.bytesToInt (b, (++pt) << 2, true); --j >= 0; ) ret[i++] = val;

}
return ret;
}, "~A,~N");
Clazz.defineMethod (c$, "rldecode32Delta", 
 function (b, n) {
if (b == null) return null;
var ret =  Clazz.newIntArray (n, 0);
for (var i = 0, pt = 0, val = 0; i < n; ) {
var diff = JU.BC.bytesToInt (b, (pt++) << 2, true);
for (var j = JU.BC.bytesToInt (b, (pt++) << 2, true); --j >= 0; ) ret[i++] = (val = val + diff);

}
return ret;
}, "~A,~N");
Clazz.defineMethod (c$, "getFloatsSplit", 
 function (xyz, factor) {
var big = this.map.get (xyz + "Big");
return (big == null ? null : this.splitDelta (big, this.map.get (xyz + "Small"), this.fileAtomCount, factor));
}, "~S,~N");
Clazz.defineMethod (c$, "splitDelta", 
 function (big, small, n, factor) {
var ret =  Clazz.newFloatArray (n, 0);
for (var i = 0, smallpt = 0, val = 0, datapt = 0, len = big.length >> 2; i < len; i++) {
ret[datapt++] = (val = val + JU.BC.bytesToInt (big, i << 2, true)) / factor;
if (++i < len) for (var j = JU.BC.bytesToInt (big, i << 2, true); --j >= 0; smallpt++) ret[datapt++] = (val = val + JU.BC.bytesToShort (small, smallpt << 1, true)) / factor;

}
return ret;
}, "~A,~A,~N,~N");
Clazz.defineMethod (c$, "getInts", 
 function (b, nbytes) {
if (b == null) return null;
var len = Clazz.doubleToInt (b.length / nbytes);
var a =  Clazz.newIntArray (len, 0);
switch (nbytes) {
case 2:
for (var i = 0, j = 0; i < len; i++, j += nbytes) a[i] = JU.BC.bytesToShort (b, j, true);

break;
case 4:
for (var i = 0, j = 0; i < len; i++, j += nbytes) a[i] = JU.BC.bytesToInt (b, j, true);

break;
}
return a;
}, "~A,~N");
Clazz.defineMethod (c$, "bytesTo4CharArray", 
 function (b) {
var id =  new Array (Clazz.doubleToInt (b.length / 4));
out : for (var i = 0, len = id.length, pt = 0; i < len; i++) {
var sb =  new JU.SB ();
for (var j = 0; j < 4; j++) {
switch (b[pt]) {
case 0:
id[i] = sb.toString ();
pt += 4 - j;
continue out;
default:
sb.appendC (String.fromCharCode (b[pt++]));
continue;
}
}
}
return id;
}, "~A");
Clazz.defineMethod (c$, "getBonds", 
 function (doMulti) {
var b = this.map.get ("bondOrderList");
var bi = this.getInts (this.map.get ("bondAtomList"), 4);
for (var i = 0, pt = 0, n = b.length; i < n; i++) {
var a1 = this.atomMap[bi[pt++]] - 1;
var a2 = this.atomMap[bi[pt++]] - 1;
if (a1 >= 0 && a2 >= 0) this.asc.addBond ( new J.adapter.smarter.Bond (a1, a2, doMulti ? b[i] : 1));
}
}, "~B");
Clazz.defineMethod (c$, "setSymmetry", 
 function () {
this.setSpaceGroupName (this.map.get ("spaceGroup"));
var o = this.map.get ("unitCell");
if (o != null) for (var i = 0; i < 6; i++) this.setUnitCellItem (i, o[i]);

});
Clazz.defineMethod (c$, "getBioAssembly", 
 function () {
var o = this.map.get ("bioAssemblyList");
if (this.vBiomolecules == null) this.vBiomolecules =  new JU.Lst ();
for (var i = o.length; --i >= 0; ) {
var info =  new java.util.Hashtable ();
this.vBiomolecules.addLast (info);
var iMolecule = i + 1;
this.checkFilterAssembly ("" + iMolecule, info);
info.put ("name", "biomolecule " + iMolecule);
info.put ("molecule", Integer.$valueOf (iMolecule));
var assemb =  new JU.Lst ();
var ops =  new JU.Lst ();
info.put ("biomts",  new JU.Lst ());
info.put ("chains",  new JU.Lst ());
info.put ("assemblies", assemb);
info.put ("operators", ops);
var m = o[i];
var tlist = m.get ("transformList");
var chlist =  new JU.SB ();
for (var j = 0, n = tlist.length; j < n; j++) {
var t = tlist[j];
chlist.setLength (0);
var chainList = t.get ("chainIndexList");
for (var k = 0, kn = chainList.length; k < kn; k++) chlist.append ("$").append (this.labelAsymList[chainList[k]]);

assemb.addLast (chlist.append ("$").toString ());
var id = "" + (++this.opCount);
this.addBiomt (id, JU.M4.newA16 (t.get ("matrix")));
ops.addLast (id);
}
}
});
Clazz.defineMethod (c$, "getAtoms", 
 function (doMulti) {
var chainsPerModel = this.map.get ("chainsPerModel");
var groupsPerChain = this.map.get ("groupsPerChain");
this.labelAsymList = this.bytesTo4CharArray (this.map.get ("chainIdList"));
var authAsymList = this.bytesTo4CharArray (this.map.get ("chainNameList"));
var groupTypeList = this.getInts (this.map.get ("groupTypeList"), 4);
var groupCount = groupTypeList.length;
this.groupModels =  Clazz.newIntArray (groupCount, 0);
var groupIdList = this.rldecode32Delta (this.map.get ("groupIdList"), groupCount);
var groupList = this.map.get ("groupList");
var insCodes = this.rldecode32 (this.map.get ("insCodeList"), groupCount);
var atomId = this.rldecode32Delta (this.map.get ("atomIdList"), this.fileAtomCount);
var haveSerial = (atomId != null);
var altloc = this.rldecode32 (this.map.get ("altLocList"), this.fileAtomCount);
var occ = this.rldecode32 (this.map.get ("occupancyList"), this.fileAtomCount);
var x = this.getFloatsSplit ("xCoord", 1000);
var y = this.getFloatsSplit ("yCoord", 1000);
var z = this.getFloatsSplit ("zCoord", 1000);
var bf = this.getFloatsSplit ("bFactor", 100);
var iatom = 0;
var nameList = (this.useAuthorChainID ? authAsymList : this.labelAsymList);
var iModel = -1;
var iChain = 0;
var nChain = 0;
var iGroup = 0;
var nGroup = 0;
var chainpt = 0;
var seqNo = 0;
var chainID = "";
var authAsym = "";
var labelAsym = "";
var insCode = 0;
this.atomMap =  Clazz.newIntArray (this.fileAtomCount, 0);
for (var j = 0; j < groupCount; j++) {
var a0 = iatom;
if (insCodes != null) insCode = insCodes[j];
seqNo = groupIdList[j];
if (++iGroup >= nGroup) {
chainID = nameList[chainpt];
authAsym = authAsymList[chainpt];
labelAsym = this.labelAsymList[chainpt];
nGroup = groupsPerChain[chainpt++];
iGroup = 0;
if (++iChain >= nChain) {
this.groupModels[j] = ++iModel;
nChain = chainsPerModel[iModel];
iChain = 0;
this.setModelPDB (true);
this.incrementModel (iModel + 1);
this.nAtoms0 = this.asc.ac;
}}var g = groupList[groupTypeList[j]];
var group3 = g.get ("groupName");
this.addHetero (group3, "" + g.get ("chemCompType"), true);
var atomNameList = g.get ("atomNameList");
var elementList = g.get ("elementList");
var len = atomNameList.length;
for (var ia = 0, pt = 0; ia < len; ia++, iatom++) {
var a =  new J.adapter.smarter.Atom ();
if (insCode != 0) a.insertionCode = String.fromCharCode (insCode);
this.setAtomCoordXYZ (a, x[iatom], y[iatom], z[iatom]);
a.elementSymbol = elementList[pt];
a.atomName = atomNameList[pt++];
if (seqNo >= 0) a.sequenceNumber = seqNo;
a.group3 = group3;
this.setChainID (a, chainID);
if (bf != null) a.bfactor = bf[iatom];
if (altloc != null) a.altLoc = String.fromCharCode (altloc[iatom]);
if (occ != null) a.foccupancy = occ[iatom] / 100;
if (haveSerial) a.atomSerial = atomId[iatom];
if (!this.filterAtom (a, -1) || !this.processSubclassAtom (a, labelAsym, authAsym)) continue;
if (haveSerial) {
this.asc.addAtomWithMappedSerialNumber (a);
} else {
this.asc.addAtom (a);
}this.atomMap[iatom] = ++this.ac;
}
if (!this.isCourseGrained) {
var bo = g.get ("bondOrderList");
if (bo != null) {
var bi = g.get ("bondAtomList");
for (var bj = 0, pt = 0, nj = bo.length; bj < nj; bj++) {
var a1 = this.atomMap[bi[pt++] + a0] - 1;
var a2 = this.atomMap[bi[pt++] + a0] - 1;
if (a1 >= 0 && a2 >= 0) this.asc.addBond ( new J.adapter.smarter.Bond (a1, a2, doMulti ? bo[bj] : 1));
}
}}}
}, "~B");
Clazz.defineMethod (c$, "getStructure", 
 function (a) {
var bsStructures =  Clazz.newArray (-1, [ new JU.BS (), null,  new JU.BS (),  new JU.BS (),  new JU.BS (), null,  new JU.BS ()]);
if (JU.Logger.debugging) JU.Logger.info (JU.PT.toJSON ("secStructList", a));
var lastGroup = -1;
for (var i = 0; i < a.length; i++) {
var type = a[i];
switch (type) {
case 0:
case 2:
case 3:
case 4:
case 6:
bsStructures[type].set (i);
lastGroup = i;
}
}
if (lastGroup >= 0) this.asc.addStructure ( new J.adapter.smarter.Structure (this.groupModels[lastGroup], null, null, null, 0, 0, bsStructures));
}, "~A");
});
