Clazz.declarePackage ("JS");
Clazz.load (["J.api.SmilesMatcherInterface"], "JS.SmilesMatcher", ["JU.AU", "$.BS", "$.PT", "JS.InvalidSmilesException", "$.SmilesAtom", "$.SmilesBond", "$.SmilesGenerator", "$.SmilesParser", "$.SmilesSearch", "JU.BNode", "$.BSUtil", "$.Elements", "$.Logger", "$.Node", "$.Point3fi"], function () {
c$ = Clazz.declareType (JS, "SmilesMatcher", null, J.api.SmilesMatcherInterface);
Clazz.overrideMethod (c$, "getLastException", 
function () {
return JS.InvalidSmilesException.getLastError ();
});
Clazz.overrideMethod (c$, "getMolecularFormula", 
function (pattern, isSmarts) {
JS.InvalidSmilesException.clear ();
var search = JS.SmilesParser.getMolecule ("/nostereo/" + pattern, isSmarts);
search.createTopoMap (null);
search.nodes = search.jmolAtoms;
return search.getMolecularFormula (!isSmarts, null, false);
}, "~S,~B");
Clazz.overrideMethod (c$, "getSmiles", 
function (atoms, ac, bsSelected, bioComment, flags) {
JS.InvalidSmilesException.clear ();
return ( new JS.SmilesGenerator ()).getSmiles (atoms, ac, bsSelected, bioComment, flags);
}, "~A,~N,JU.BS,~S,~N");
Clazz.overrideMethod (c$, "areEqual", 
function (smiles1, smiles2) {
JS.InvalidSmilesException.clear ();
var result = this.findPriv (smiles1, JS.SmilesParser.getMolecule (smiles2, false), (smiles1.indexOf ("*") >= 0 ? 2 : 1) | 32, 2);
return (result == null ? -1 : result.length);
}, "~S,~S");
Clazz.defineMethod (c$, "areEqualTest", 
function (smiles, molecule) {
var ret = this.findPriv (smiles, molecule, 33, 2);
return (ret != null && ret.length == 1);
}, "~S,JS.SmilesSearch");
Clazz.overrideMethod (c$, "find", 
function (pattern, smiles, isSmarts, firstMatchOnly) {
JS.InvalidSmilesException.clear ();
smiles = JS.SmilesParser.cleanPattern (smiles);
pattern = JS.SmilesParser.cleanPattern (pattern);
var search = JS.SmilesParser.getMolecule (smiles, false);
var array = this.findPriv (pattern, search, (isSmarts ? 2 : 1) | (firstMatchOnly ? 32 : 0), 3);
for (var i = array.length; --i >= 0; ) {
var a = array[i];
for (var j = a.length; --j >= 0; ) a[j] = (search.jmolAtoms[a[j]]).mapIndex;

}
return array;
}, "~S,~S,~B,~B");
Clazz.overrideMethod (c$, "getRelationship", 
function (smiles1, smiles2) {
if (smiles1 == null || smiles2 == null || smiles1.length == 0 || smiles2.length == 0) return "";
var mf1 = this.getMolecularFormula (smiles1, false);
var mf2 = this.getMolecularFormula (smiles2, false);
if (!mf1.equals (mf2)) return "none";
var check;
var n1 = JU.PT.countChar (JU.PT.rep (smiles1, "@@", "@"), '@');
var n2 = JU.PT.countChar (JU.PT.rep (smiles2, "@@", "@"), '@');
check = (n1 == n2 && this.areEqual (smiles2, smiles1) > 0);
if (!check) {
var s = smiles1 + smiles2;
if (s.indexOf ("/") >= 0 || s.indexOf ("\\") >= 0 || s.indexOf ("@") >= 0) {
if (n1 == n2 && n1 > 0 && s.indexOf ("@SP") < 0) {
check = (this.areEqual ("/invertstereo/" + smiles2, smiles1) > 0);
if (check) return "enantiomers";
}check = (this.areEqual ("/nostereo/" + smiles2, smiles1) > 0);
if (check) return (n1 == n2 ? "diastereomers" : "ambiguous stereochemistry!");
}return "constitutional isomers";
}return "identical";
}, "~S,~S");
Clazz.overrideMethod (c$, "reverseChirality", 
function (smiles) {
smiles = JU.PT.rep (smiles, "@@", "!@");
smiles = JU.PT.rep (smiles, "@", "@@");
smiles = JU.PT.rep (smiles, "!@@", "@");
return smiles;
}, "~S");
Clazz.overrideMethod (c$, "getSubstructureSet", 
function (pattern, atoms, ac, bsSelected, flags) {
return this.matchPriv (pattern, atoms, ac, bsSelected, null, true, flags, 1);
}, "~S,~A,~N,JU.BS,~N");
Clazz.overrideMethod (c$, "getMMFF94AtomTypes", 
function (smarts, atoms, ac, bsSelected, ret, vRings) {
JS.InvalidSmilesException.clear ();
var sp =  new JS.SmilesParser (true);
var search = null;
var flags = (1794);
search = sp.parse ("");
search.exitFirstMatch = false;
search.jmolAtoms = atoms;
search.jmolAtomCount = Math.abs (ac);
search.setSelected (bsSelected);
search.flags = flags;
search.getRingData (vRings, true, true);
search.asVector = false;
search.subSearches =  new Array (1);
search.getSelections ();
var bsDone =  new JU.BS ();
for (var i = 0; i < smarts.length; i++) {
if (smarts[i] == null || smarts[i].length == 0 || smarts[i].startsWith ("#")) {
ret.addLast (null);
continue;
}search.clear ();
var ss = sp.getSearch (search, JS.SmilesParser.cleanPattern (smarts[i]), flags);
search.subSearches[0] = ss;
var bs = JU.BSUtil.copy (search.search ());
ret.addLast (bs);
bsDone.or (bs);
if (bsDone.cardinality () == ac) return;
}
}, "~A,~A,~N,JU.BS,JU.Lst,~A");
Clazz.overrideMethod (c$, "getSubstructureSetArray", 
function (pattern, atoms, ac, bsSelected, bsAromatic, flags) {
return this.matchPriv (pattern, atoms, ac, bsSelected, bsAromatic, true, flags, 2);
}, "~S,~A,~N,JU.BS,JU.BS,~N");
Clazz.overrideMethod (c$, "polyhedronToSmiles", 
function (center, faces, atomCount, points, flags, details) {
var atoms =  new Array (atomCount);
for (var i = 0; i < atomCount; i++) {
atoms[i] =  new JS.SmilesAtom ();
var pt = (points == null ? null : points[i]);
if (Clazz.instanceOf (pt, JU.Node)) {
atoms[i].elementNumber = (pt).getElementNumber ();
atoms[i].atomName = (pt).getAtomName ();
atoms[i].atomNumber = (pt).getAtomNumber ();
atoms[i].setT (pt);
} else {
atoms[i].elementNumber = (Clazz.instanceOf (pt, JU.Point3fi) ? (pt).sD : -2);
}atoms[i].index = i;
}
var nBonds = 0;
for (var i = faces.length; --i >= 0; ) {
var face = faces[i];
var n = face.length;
var iatom;
var iatom2;
for (var j = n; --j >= 0; ) {
if ((iatom = face[j]) >= atomCount || (iatom2 = face[(j + 1) % n]) >= atomCount) continue;
if (atoms[iatom].getBondTo (atoms[iatom2]) == null) {
var b =  new JS.SmilesBond (atoms[iatom], atoms[iatom2], 1, false);
b.index = nBonds++;
}}
}
for (var i = 0; i < atomCount; i++) {
var n = atoms[i].bondCount;
if (n == 0 || n != atoms[i].bonds.length) atoms[i].bonds = JU.AU.arrayCopyObject (atoms[i].bonds, n);
}
var s = null;
var g =  new JS.SmilesGenerator ();
if (points != null) g.stereoReference = center;
JS.InvalidSmilesException.clear ();
s = g.getSmiles (atoms, atomCount, JU.BSUtil.newBitSet2 (0, atomCount), null, flags | 4096 | 16384 | 32768);
if ((flags & 65536) == 65536) {
s = "//* " + center + " *//\t[" + JU.Elements.elementSymbolFromNumber (center.getElementNumber ()) + "@PH" + atomCount + (details == null ? "" : "/" + details + "/") + "]." + s;
}return s;
}, "JU.Node,~A,~N,~A,~N,~S");
Clazz.overrideMethod (c$, "getCorrelationMaps", 
function (pattern, atoms, atomCount, bsSelected, flags) {
return this.matchPriv (pattern, atoms, atomCount, bsSelected, null, true, flags, 3);
}, "~S,~A,~N,JU.BS,~N");
Clazz.defineMethod (c$, "findPriv", 
 function (pattern, search, flags, mode) {
var bsAromatic =  new JU.BS ();
search.setFlags (search.flags | JS.SmilesParser.getFlags (pattern));
search.createTopoMap (bsAromatic);
return this.matchPriv (pattern, search.jmolAtoms, -search.jmolAtoms.length, null, bsAromatic, bsAromatic.isEmpty (), flags, mode);
}, "~S,JS.SmilesSearch,~N,~N");
Clazz.defineMethod (c$, "matchPriv", 
 function (pattern, atoms, ac, bsSelected, bsAromatic, doTestAromatic, flags, mode) {
JS.InvalidSmilesException.clear ();
try {
var isSmarts = ((flags & 2) == 2);
var search = JS.SmilesParser.getMolecule (pattern, isSmarts);
if (search.openSMILES && !isSmarts && !search.patternAromatic) JS.SmilesSearch.normalizeAromaticity (search.patternAtoms, bsAromatic, search.flags);
search.jmolAtoms = atoms;
search.jmolAtomCount = Math.abs (ac);
if (ac < 0) search.isSmilesFind = true;
var is3D = !(Clazz.instanceOf (atoms[0], JS.SmilesAtom));
if (Clazz.instanceOf (atoms[0], JU.BNode)) search.bioAtoms = atoms;
search.setSelected (bsSelected);
search.getSelections ();
search.bsRequired = null;
if (!doTestAromatic) search.bsAromatic = bsAromatic;
search.setRingData (null, null, is3D || doTestAromatic);
search.exitFirstMatch = ((flags & 32) == 32);
switch (mode) {
case 1:
search.asVector = false;
return search.search ();
case 2:
search.asVector = true;
var vb = search.search ();
return vb.toArray ( new Array (vb.size ()));
case 3:
search.getMaps = true;
var vl = search.search ();
return vl.toArray (JU.AU.newInt2 (vl.size ()));
}
} catch (e) {
if (Clazz.exceptionOf (e, Exception)) {
if (JU.Logger.debugging) e.printStackTrace ();
if (JS.InvalidSmilesException.getLastError () == null) JS.InvalidSmilesException.clear ();
throw  new JS.InvalidSmilesException (JS.InvalidSmilesException.getLastError ());
} else {
throw e;
}
}
return null;
}, "~S,~A,~N,JU.BS,JU.BS,~B,~N,~N");
Clazz.overrideMethod (c$, "cleanSmiles", 
function (smiles) {
return JS.SmilesParser.cleanPattern (smiles);
}, "~S");
Clazz.defineStatics (c$,
"MODE_BITSET", 0x01,
"MODE_ARRAY", 0x02,
"MODE_MAP", 0x03);
});
