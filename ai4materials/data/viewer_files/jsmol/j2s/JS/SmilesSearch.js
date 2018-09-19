Clazz.declarePackage ("JS");
Clazz.load (["JU.JmolMolecule", "JU.BS", "$.Lst", "JS.VTemp"], "JS.SmilesSearch", ["java.lang.Float", "java.util.Arrays", "$.Hashtable", "JU.AU", "$.SB", "JS.SmilesAromatic", "$.SmilesAtom", "$.SmilesBond", "$.SmilesMeasure", "$.SmilesParser", "JU.BSUtil", "$.Logger"], function () {
c$ = Clazz.decorateAsClass (function () {
this.patternAtoms = null;
this.pattern = null;
this.jmolAtoms = null;
this.v = null;
this.bioAtoms = null;
this.jmolAtomCount = 0;
this.bsSelected = null;
this.bsRequired = null;
this.exitFirstMatch = false;
this.isSmarts = false;
this.isSmilesFind = false;
this.isTopology = false;
this.patternAromatic = true;
this.subSearches = null;
this.haveSelected = false;
this.haveBondStereochemistry = false;
this.stereo = null;
this.needRingData = false;
this.needAromatic = true;
this.needRingMemberships = false;
this.nDouble = 0;
this.ringDataMax = -2147483648;
this.ringSets = null;
this.ringCount = 0;
this.measures = null;
this.flags = 0;
this.bsAromatic = null;
this.bsAromatic5 = null;
this.bsAromatic6 = null;
this.lastChainAtom = null;
this.asVector = false;
this.getMaps = false;
this.top = null;
this.isSilent = false;
this.isRingCheck = false;
this.selectedAtomCount = 0;
this.ringData = null;
this.ringCounts = null;
this.ringConnections = null;
this.bsFound = null;
this.htNested = null;
this.nNested = 0;
this.nestedBond = null;
this.vReturn = null;
this.bsReturn = null;
this.ignoreStereochemistry = false;
this.invertStereochemistry = false;
this.noAromatic = false;
this.aromaticDouble = false;
this.jsmeNoncanonical = false;
this.openSMILES = false;
this.bsCheck = null;
Clazz.instantialize (this, arguments);
}, JS, "SmilesSearch", JU.JmolMolecule);
Clazz.prepareFields (c$, function () {
this.patternAtoms =  new Array (16);
this.v =  new JS.VTemp ();
this.measures =  new JU.Lst ();
this.bsAromatic =  new JU.BS ();
this.bsAromatic5 =  new JU.BS ();
this.bsAromatic6 =  new JU.BS ();
this.top = this;
this.bsFound =  new JU.BS ();
this.bsReturn =  new JU.BS ();
});
Clazz.makeConstructor (c$, 
function () {
Clazz.superConstructor (this, JS.SmilesSearch, []);
});
Clazz.defineMethod (c$, "toString", 
function () {
var sb =  new JU.SB ().append (this.pattern);
sb.append ("\nmolecular formula: " + this.getMolecularFormula (true, null, false));
return sb.toString ();
});
c$.addFlags = Clazz.defineMethod (c$, "addFlags", 
function (flags, strFlags) {
if (strFlags.indexOf ("OPEN") >= 0) flags |= 5;
if (strFlags.indexOf ("BIO") >= 0) flags |= 1048576;
if (strFlags.indexOf ("NONCANONICAL") >= 0) flags |= 2048;
if (strFlags.indexOf ("STRICT") >= 0) flags |= 256;
if (strFlags.indexOf ("NOAROMATIC") >= 0 || strFlags.indexOf ("NONAROMATIC") >= 0) flags |= 16;
if (strFlags.indexOf ("AROMATICDOUBLE") >= 0) flags |= 512;
if (strFlags.indexOf ("AROMATICDEFINED") >= 0) flags |= 128;
if (strFlags.indexOf ("MMFF94") >= 0) flags |= 1792;
if (strFlags.indexOf ("NOSTEREO") >= 0) {
flags |= 32;
} else if (strFlags.indexOf ("INVERTSTEREO") >= 0) {
if ((flags & 64) != 0) flags &= -65;
 else flags |= 64;
}if ((flags & 1048576) == 1048576) {
if (strFlags.indexOf ("NOCOMMENT") >= 0) flags |= 34603008;
if (strFlags.indexOf ("ATOMCOMMENT") >= 0) flags |= 131072;
if (strFlags.indexOf ("UNMATCHED") >= 0) flags |= 3145728;
if (strFlags.indexOf ("COVALENT") >= 0) flags |= 5242880;
if (strFlags.indexOf ("HBOND") >= 0) flags |= 9437184;
}return flags;
}, "~N,~S");
Clazz.defineMethod (c$, "setFlags", 
function (flags) {
this.flags = flags;
this.noAromatic = ((flags & 16) == 16);
this.aromaticDouble = ((flags & 512) == 512);
this.jsmeNoncanonical = ((flags & 2048) == 2048);
this.ignoreStereochemistry = ((flags & 32) == 32);
this.invertStereochemistry = ((flags & 64) == 64);
this.openSMILES = ((flags & 5) == 5);
}, "~N");
Clazz.defineMethod (c$, "setSelected", 
function (bs) {
if (bs == null) {
bs = JU.BS.newN (this.jmolAtomCount);
bs.setBits (0, this.jmolAtomCount);
}this.bsSelected = bs;
}, "JU.BS");
Clazz.defineMethod (c$, "setAtomArray", 
function () {
if (this.patternAtoms.length > this.ac) this.patternAtoms = JU.AU.arrayCopyObject (this.patternAtoms, this.ac);
this.nodes = this.patternAtoms;
});
Clazz.defineMethod (c$, "addAtom", 
function () {
return this.appendAtom ( new JS.SmilesAtom ());
});
Clazz.defineMethod (c$, "appendAtom", 
function (sAtom) {
if (this.ac >= this.patternAtoms.length) this.patternAtoms = JU.AU.doubleLength (this.patternAtoms);
return this.patternAtoms[this.ac] = sAtom.setIndex (this.ac++);
}, "JS.SmilesAtom");
Clazz.defineMethod (c$, "addNested", 
function (pattern) {
if (this.top.htNested == null) this.top.htNested =  new java.util.Hashtable ();
this.setNested (++this.top.nNested, pattern);
return this.top.nNested;
}, "~S");
Clazz.defineMethod (c$, "clear", 
function () {
this.bsReturn.clearAll ();
this.nNested = 0;
this.htNested = null;
this.nestedBond = null;
this.clearBsFound (-1);
});
Clazz.defineMethod (c$, "setNested", 
function (iNested, o) {
this.top.htNested.put ("_" + iNested, o);
}, "~N,~O");
Clazz.defineMethod (c$, "getNested", 
function (iNested) {
return this.top.htNested.get ("_" + iNested);
}, "~N");
Clazz.defineMethod (c$, "getMissingHydrogenCount", 
function () {
var n = 0;
var nH;
for (var i = 0; i < this.ac; i++) if ((nH = this.patternAtoms[i].explicitHydrogenCount) >= 0) n += nH;

return n;
});
Clazz.defineMethod (c$, "setRingData", 
function (bsA, vRings, doProcessAromatic) {
if (this.isTopology) this.needAromatic = false;
if (this.needAromatic) this.needRingData = true;
var noAromatic = ((this.flags & 16) == 16);
this.needAromatic = new Boolean (this.needAromatic & ( new Boolean ((bsA == null) & !noAromatic).valueOf ())).valueOf ();
if (!this.needAromatic) {
this.bsAromatic.clearAll ();
if (bsA != null) this.bsAromatic.or (bsA);
if (!this.needRingMemberships && !this.needRingData) return;
}this.getRingData (vRings, this.needRingData, doProcessAromatic);
}, "JU.BS,~A,~B");
Clazz.defineMethod (c$, "getRingData", 
function (vRings, needRingData, doTestAromatic) {
var isStrict = ((this.flags & 256) == 256);
var strictness = (!isStrict ? 0 : (this.flags & 1792) == 1792 ? 2 : 1);
var isDefined = ((this.flags & 128) == 128);
var isOpenNotStrict = (!isStrict && (this.flags & 5) == 5);
var doFinalize = (this.needAromatic && doTestAromatic && (isStrict || isOpenNotStrict));
var aromaticMax = 7;
var lstAromatic = (vRings == null ?  new JU.Lst () : (vRings[3] =  new JU.Lst ()));
var lstSP2 = (doFinalize ?  new JU.Lst () : null);
var eCounts = (doFinalize ?  Clazz.newIntArray (this.jmolAtomCount, 0) : null);
if (isDefined && this.needAromatic) {
JS.SmilesAromatic.checkAromaticDefined (this.jmolAtoms, this.bsSelected, this.bsAromatic);
strictness = 0;
}var nAtoms = this.jmolAtomCount;
var checkFlatness = (nAtoms > 0 && !(Clazz.instanceOf (this.jmolAtoms[0], JS.SmilesAtom)));
if (this.ringDataMax < 0) this.ringDataMax = 8;
if (strictness > 0 && this.ringDataMax < 6) this.ringDataMax = 6;
if (needRingData) {
this.ringCounts =  Clazz.newIntArray (nAtoms, 0);
this.ringConnections =  Clazz.newIntArray (this.jmolAtomCount, 0);
this.ringData =  new Array (this.ringDataMax + 1);
}this.ringSets =  new JU.SB ();
var s = "****";
var max = this.ringDataMax;
while (s.length < max) s += s;

for (var i = 3; i <= max; i++) {
if (i > nAtoms) continue;
var smarts = "*1" + s.substring (0, i - 2) + "*1";
var search = JS.SmilesParser.getMolecule (smarts, true);
var vR = this.subsearch (search, false, true);
if (vRings != null && i <= 5) {
var v =  new JU.Lst ();
for (var j = vR.size (); --j >= 0; ) v.addLast (vR.get (j));

vRings[i - 3] = v;
}if (vR.size () == 0) continue;
if (this.needAromatic && !isDefined && i >= 5 && i <= aromaticMax) JS.SmilesAromatic.setAromatic (i, this.jmolAtoms, this.bsSelected, vR, this.bsAromatic, strictness, isOpenNotStrict, checkFlatness, this.v, lstAromatic, lstSP2, eCounts, doTestAromatic);
if (needRingData) {
this.ringData[i] =  new JU.BS ();
for (var k = vR.size (); --k >= 0; ) {
var r = vR.get (k);
this.ringData[i].or (r);
for (var j = r.nextSetBit (0); j >= 0; j = r.nextSetBit (j + 1)) this.ringCounts[j]++;

}
}}
if (this.needAromatic) {
if (doFinalize) JS.SmilesAromatic.finalizeAromatic (this.jmolAtoms, this.bsAromatic, lstAromatic, lstSP2, eCounts, isOpenNotStrict, isStrict);
this.bsAromatic5.clearAll ();
this.bsAromatic6.clearAll ();
for (var i = lstAromatic.size (); --i >= 0; ) {
var bs = lstAromatic.get (i);
bs.and (this.bsAromatic);
switch (bs.cardinality ()) {
case 5:
this.bsAromatic5.or (bs);
break;
case 6:
this.bsAromatic6.or (bs);
break;
}
}
}if (needRingData) {
for (var i = this.bsSelected.nextSetBit (0); i >= 0; i = this.bsSelected.nextSetBit (i + 1)) {
var atom = this.jmolAtoms[i];
var bonds = atom.getEdges ();
if (bonds != null) for (var k = bonds.length; --k >= 0; ) if (this.ringCounts[atom.getBondedAtomIndex (k)] > 0) this.ringConnections[i]++;

}
}}, "~A,~B,~B");
Clazz.defineMethod (c$, "subsearch", 
function (search, firstAtomOnly, isRingCheck) {
search.ringSets = this.ringSets;
search.jmolAtoms = this.jmolAtoms;
search.bioAtoms = this.bioAtoms;
search.jmolAtomCount = this.jmolAtomCount;
search.bsSelected = this.bsSelected;
search.htNested = this.htNested;
search.isSmilesFind = this.isSmilesFind;
search.bsCheck = this.bsCheck;
search.isSmarts = true;
search.bsAromatic = this.bsAromatic;
search.bsAromatic5 = this.bsAromatic5;
search.bsAromatic6 = this.bsAromatic6;
search.ringData = this.ringData;
search.ringCounts = this.ringCounts;
search.ringConnections = this.ringConnections;
if (firstAtomOnly) {
search.bsRequired = null;
search.exitFirstMatch = false;
} else if (isRingCheck) {
search.bsRequired = null;
search.isSilent = true;
search.isRingCheck = true;
search.asVector = true;
} else {
search.haveSelected = this.haveSelected;
search.bsRequired = this.bsRequired;
search.exitFirstMatch = this.exitFirstMatch;
search.getMaps = this.getMaps;
search.asVector = this.asVector;
search.vReturn = this.vReturn;
search.bsReturn = this.bsReturn;
}return search.search2 (firstAtomOnly);
}, "JS.SmilesSearch,~B,~B");
Clazz.defineMethod (c$, "search", 
function () {
return this.search2 (false);
});
Clazz.defineMethod (c$, "search2", 
 function (firstAtomOnly) {
this.setFlags (this.flags);
if (JU.Logger.debugging && !this.isSilent) JU.Logger.debug ("SmilesSearch processing " + this.pattern);
if (this.vReturn == null && (this.asVector || this.getMaps)) this.vReturn =  new JU.Lst ();
if (this.bsSelected == null) {
this.bsSelected = JU.BS.newN (this.jmolAtomCount);
this.bsSelected.setBits (0, this.jmolAtomCount);
}this.selectedAtomCount = this.bsSelected.cardinality ();
if (this.subSearches != null) {
for (var i = 0; i < this.subSearches.length; i++) {
if (this.subSearches[i] == null) continue;
this.subsearch (this.subSearches[i], false, false);
if (this.exitFirstMatch) {
if (this.vReturn == null ? this.bsReturn.nextSetBit (0) >= 0 : this.vReturn.size () > 0) break;
}}
} else if (this.ac > 0) {
this.checkMatch (null, -1, -1, firstAtomOnly);
}return (this.asVector || this.getMaps ? this.vReturn : this.bsReturn);
}, "~B");
Clazz.defineMethod (c$, "checkMatch", 
 function (patternAtom, atomNum, iAtom, firstAtomOnly) {
var jmolAtom;
var jmolBonds;
if (patternAtom == null) {
if (this.nestedBond == null) {
this.clearBsFound (-1);
} else {
this.bsReturn.clearAll ();
}} else {
if (this.bsFound.get (iAtom) || !this.bsSelected.get (iAtom)) return true;
jmolAtom = this.jmolAtoms[iAtom];
if (!this.isRingCheck && !this.isTopology) {
if (patternAtom.atomsOr != null) {
for (var ii = 0; ii < patternAtom.nAtomsOr; ii++) if (!this.checkMatch (patternAtom.atomsOr[ii], atomNum, iAtom, firstAtomOnly)) return false;

return true;
}if (patternAtom.primitives == null) {
if (!this.checkPrimitiveAtom (patternAtom, iAtom)) return true;
} else {
for (var i = 0; i < patternAtom.nPrimitives; i++) if (!this.checkPrimitiveAtom (patternAtom.primitives[i], iAtom)) return true;

}}jmolBonds = jmolAtom.getEdges ();
for (var i = patternAtom.getBondCount (); --i >= 0; ) {
var patternBond = patternAtom.getBond (i);
if (patternBond.getAtomIndex2 () != patternAtom.index) continue;
var atom1 = patternBond.atom1;
var matchingAtom = atom1.getMatchingAtomIndex ();
switch (patternBond.order) {
case 96:
case 112:
if (!this.checkMatchBond (patternAtom, atom1, patternBond, iAtom, matchingAtom, null)) return true;
break;
default:
var k = 0;
for (; k < jmolBonds.length; k++) if ((jmolBonds[k].getAtomIndex1 () == matchingAtom || jmolBonds[k].getAtomIndex2 () == matchingAtom) && jmolBonds[k].isCovalent ()) break;

if (k == jmolBonds.length) return true;
if (!this.checkMatchBond (patternAtom, atom1, patternBond, iAtom, matchingAtom, jmolBonds[k])) return true;
}
}
this.patternAtoms[patternAtom.index].setMatchingAtom (this.jmolAtoms[iAtom], iAtom);
if (JU.Logger.debugging && !this.isSilent) {
for (var i = 0; i <= atomNum; i++) JU.Logger.debug ("pattern atoms " + this.patternAtoms[i]);

JU.Logger.debug ("--ss--");
}this.bsFound.set (iAtom);
}if (!this.continueMatch (atomNum, iAtom, firstAtomOnly)) return false;
if (iAtom >= 0) this.clearBsFound (iAtom);
return true;
}, "JS.SmilesAtom,~N,~N,~B");
Clazz.defineMethod (c$, "continueMatch", 
 function (atomNum, iAtom, firstAtomOnly) {
var jmolAtom;
var jmolBonds;
if (++atomNum < this.ac) {
var newPatternAtom = this.patternAtoms[atomNum];
var newPatternBond = (iAtom >= 0 ? newPatternAtom.getBondTo (null) : atomNum == 0 ? this.nestedBond : null);
if (newPatternBond == null) {
var bs = JU.BSUtil.copy (this.bsFound);
var bs0 = JU.BSUtil.copy (this.bsFound);
if (newPatternAtom.notBondedIndex >= 0) {
var pa = this.patternAtoms[newPatternAtom.notBondedIndex];
var a = pa.getMatchingAtom ();
if (pa.isBioAtom) {
var ii = (a).getOffsetResidueAtom ("\0", 1);
if (ii >= 0) bs.set (ii);
ii = (a).getOffsetResidueAtom ("\0", -1);
if (ii >= 0) bs.set (ii);
} else {
jmolBonds = a.getEdges ();
for (var k = 0; k < jmolBonds.length; k++) bs.set (jmolBonds[k].getOtherAtomNode (a).getIndex ());

}}var skipGroup = (iAtom >= 0 && newPatternAtom.isBioAtom && (newPatternAtom.atomName == null || newPatternAtom.residueChar != null));
for (var j = this.bsSelected.nextSetBit (0); j >= 0; j = this.bsSelected.nextSetBit (j + 1)) {
if (!bs.get (j) && !this.checkMatch (newPatternAtom, atomNum, j, firstAtomOnly)) return false;
if (skipGroup) {
var j1 = (this.jmolAtoms[j]).getOffsetResidueAtom (newPatternAtom.atomName, 1);
if (j1 >= 0) j = j1 - 1;
}}
this.bsFound = bs0;
return true;
}jmolAtom = newPatternBond.atom1.getMatchingAtom ();
switch (newPatternBond.order) {
case 96:
var nextGroupAtom = (jmolAtom).getOffsetResidueAtom (newPatternAtom.atomName, 1);
if (nextGroupAtom >= 0) {
var bs = JU.BSUtil.copy (this.bsFound);
(jmolAtom).getGroupBits (this.bsFound);
if (!this.checkMatch (newPatternAtom, atomNum, nextGroupAtom, firstAtomOnly)) return false;
this.bsFound = bs;
}return true;
case 112:
var vLinks =  new JU.Lst ();
(jmolAtom).getCrossLinkVector (vLinks, true, true);
var bs = JU.BSUtil.copy (this.bsFound);
(jmolAtom).getGroupBits (this.bsFound);
for (var j = 2; j < vLinks.size (); j += 3) if (!this.checkMatch (newPatternAtom, atomNum, vLinks.get (j).intValue (), firstAtomOnly)) return false;

this.bsFound = bs;
return true;
}
jmolBonds = jmolAtom.getEdges ();
if (jmolBonds != null) for (var j = 0; j < jmolBonds.length; j++) if (!this.checkMatch (newPatternAtom, atomNum, jmolAtom.getBondedAtomIndex (j), firstAtomOnly)) return false;

this.clearBsFound (iAtom);
return true;
}if (!this.ignoreStereochemistry && !this.checkStereochemistry ()) return true;
var bs =  new JU.BS ();
var nMatch = 0;
for (var j = 0; j < this.ac; j++) {
var i = this.patternAtoms[j].getMatchingAtomIndex ();
if (!firstAtomOnly && this.top.haveSelected && !this.patternAtoms[j].selected) continue;
nMatch++;
bs.set (i);
if (this.patternAtoms[j].isBioAtom && this.patternAtoms[j].atomName == null) (this.jmolAtoms[i]).getGroupBits (bs);
if (firstAtomOnly) break;
if (!this.isSmarts && this.patternAtoms[j].explicitHydrogenCount > 0) this.getHydrogens (this.jmolAtoms[i], bs);
}
if (this.bsRequired != null && !this.bsRequired.intersects (bs)) return true;
if (!this.isSmarts && bs.cardinality () != this.selectedAtomCount) return true;
if (this.bsCheck != null) {
if (firstAtomOnly) {
this.bsCheck.clearAll ();
for (var j = 0; j < this.ac; j++) {
this.bsCheck.set (this.patternAtoms[j].getMatchingAtomIndex ());
}
if (this.bsCheck.cardinality () != this.ac) return true;
} else {
if (bs.cardinality () != this.ac) return true;
}}this.bsReturn.or (bs);
if (this.getMaps) {
var map =  Clazz.newIntArray (nMatch, 0);
for (var j = 0, nn = 0; j < this.ac; j++) {
if (!firstAtomOnly && this.top.haveSelected && !this.patternAtoms[j].selected) continue;
map[nn++] = this.patternAtoms[j].getMatchingAtomIndex ();
}
this.vReturn.addLast (map);
return !this.exitFirstMatch;
}if (this.asVector) {
var isOK = true;
for (var j = this.vReturn.size (); --j >= 0 && isOK; ) isOK = !((this.vReturn.get (j)).equals (bs));

if (!isOK) return true;
this.vReturn.addLast (bs);
}if (this.isRingCheck) {
this.ringSets.append (" ");
for (var k = atomNum * 3 + 2; --k > atomNum; ) this.ringSets.append ("-").appendI (this.patternAtoms[(k <= atomNum * 2 ? atomNum * 2 - k + 1 : k - 1) % atomNum].getMatchingAtomIndex ());

this.ringSets.append ("- ");
return true;
}if (this.exitFirstMatch) return false;
return (bs.cardinality () != this.selectedAtomCount);
}, "~N,~N,~B");
Clazz.defineMethod (c$, "clearBsFound", 
 function (iAtom) {
if (iAtom < 0) {
if (this.bsCheck == null) {
this.bsFound.clearAll ();
}} else this.bsFound.clear (iAtom);
}, "~N");
Clazz.defineMethod (c$, "getHydrogens", 
function (atom, bsHydrogens) {
var b = atom.getEdges ();
var k = -1;
for (var i = 0; i < b.length; i++) if (this.jmolAtoms[atom.getBondedAtomIndex (i)].getElementNumber () == 1) {
k = atom.getBondedAtomIndex (i);
if (bsHydrogens == null) break;
bsHydrogens.set (k);
}
return (k >= 0 ? this.jmolAtoms[k] : null);
}, "JU.Node,JU.BS");
Clazz.defineMethod (c$, "checkPrimitiveAtom", 
 function (patternAtom, iAtom) {
var atom = this.jmolAtoms[iAtom];
var foundAtom = patternAtom.not;
while (true) {
var n;
if (patternAtom.iNested > 0) {
var o = this.getNested (patternAtom.iNested);
if (Clazz.instanceOf (o, JS.SmilesSearch)) {
var search = o;
if (patternAtom.isBioAtom) search.nestedBond = patternAtom.getBondTo (null);
o = this.subsearch (search, true, false);
if (o == null) o =  new JU.BS ();
if (!patternAtom.isBioAtom) this.setNested (patternAtom.iNested, o);
}foundAtom = (patternAtom.not != ((o).get (iAtom)));
break;
}if (patternAtom.atomNumber != -2147483648 && patternAtom.atomNumber != ((atom).getAtomNumber ())) break;
if (patternAtom.elementNumber >= 0 && patternAtom.elementNumber != atom.getElementNumber ()) break;
if (patternAtom.jmolIndex >= 0 && atom.getIndex () != patternAtom.jmolIndex) break;
if (patternAtom.atomName != null && (patternAtom.isLeadAtom () ? !(atom).isLeadAtom () : !patternAtom.atomName.equals ((atom).getAtomName ().toUpperCase ()))) break;
if (patternAtom.isBioResidue) {
var a = atom;
if (patternAtom.residueName != null && !patternAtom.residueName.equals (a.getGroup3 (false).toUpperCase ())) break;
if (patternAtom.residueNumber != -2147483648 && patternAtom.residueNumber != (a.getResno ())) break;
if (patternAtom.residueChar != null || patternAtom.elementNumber == -2) {
var atype = a.getBioSmilesType ();
var ptype = patternAtom.getBioSmilesType ();
var ok = true;
var isNucleic = false;
switch (ptype) {
case '\0':
case '*':
ok = true;
break;
case 'n':
ok = (atype == 'r' || atype == 'c');
isNucleic = true;
break;
case 'r':
case 'c':
isNucleic = true;
default:
ok = (atype == ptype);
break;
}
if (!ok) break;
var s = a.getGroup1 ('\0').toUpperCase ();
var resChar = (patternAtom.residueChar == null ? '*' : patternAtom.residueChar.charAt (0));
var isOK = (resChar == s.charAt (0));
switch (resChar) {
case '*':
isOK = true;
break;
case 'N':
isOK = isNucleic ? (atype == 'r' || atype == 'c') : isOK;
break;
case 'R':
isOK = isNucleic ? a.isPurine () : isOK;
break;
case 'Y':
isOK = isNucleic ? a.isPyrimidine () : isOK;
break;
}
if (!isOK) break;
}if (patternAtom.isBioAtom) {
if (patternAtom.notCrossLinked && a.getCrossLinkVector (null, true, true)) break;
}} else {
if (patternAtom.atomType != null && !patternAtom.atomType.equals (atom.getAtomType ())) break;
var isAromatic = patternAtom.isAromatic ();
if (!this.noAromatic && !patternAtom.aromaticAmbiguous && isAromatic != this.bsAromatic.get (iAtom)) {
if (!this.jsmeNoncanonical || patternAtom.getExplicitHydrogenCount () != atom.getCovalentHydrogenCount ()) break;
}if ((n = patternAtom.getAtomicMass ()) != -2147483648) {
var isotope = atom.getIsotopeNumber ();
if (n >= 0 && n != isotope || n < 0 && isotope != 0 && -n != isotope) {
break;
}}if ((n = patternAtom.getCharge ()) != -2147483648 && n != atom.getFormalCharge ()) break;
n = patternAtom.getCovalentHydrogenCount () + patternAtom.explicitHydrogenCount;
if (n >= 0 && n != atom.getCovalentHydrogenCount ()) break;
n = patternAtom.implicitHydrogenCount;
if (n != -2147483648) {
var nH = atom.getImplicitHydrogenCount ();
if (n == -1 ? nH == 0 : n != nH) break;
}if (patternAtom.degree > 0 && patternAtom.degree != atom.getCovalentBondCount ()) break;
if (patternAtom.nonhydrogenDegree > 0 && patternAtom.nonhydrogenDegree != atom.getCovalentBondCount () - atom.getCovalentHydrogenCount ()) break;
if (this.isSmarts && patternAtom.valence > 0 && patternAtom.valence != atom.getValence ()) break;
if (patternAtom.connectivity > 0 && patternAtom.connectivity != atom.getCovalentBondCount () + atom.getImplicitHydrogenCount ()) break;
if (this.openSMILES) {
if (!Float.isNaN (patternAtom.atomClass) && patternAtom.atomClass != Clazz.floatToInt (atom.getFloatProperty ("property_atomclass"))) break;
}if (this.ringData != null) {
if (patternAtom.ringSize >= -1) {
if (patternAtom.ringSize <= 0) {
if ((this.ringCounts[iAtom] == 0) != (patternAtom.ringSize == 0)) break;
} else {
var rd = this.ringData[patternAtom.ringSize == 500 ? 5 : patternAtom.ringSize == 600 ? 6 : patternAtom.ringSize];
if (rd == null || !rd.get (iAtom)) break;
if (!this.noAromatic) if (patternAtom.ringSize == 500) {
if (!this.bsAromatic5.get (iAtom)) break;
} else if (patternAtom.ringSize == 600) {
if (!this.bsAromatic6.get (iAtom)) break;
}}}if (patternAtom.ringMembership >= -1) {
if (patternAtom.ringMembership == -1 ? this.ringCounts[iAtom] == 0 : this.ringCounts[iAtom] != patternAtom.ringMembership) break;
}if (patternAtom.ringConnectivity >= 0) {
n = this.ringConnections[iAtom];
if (patternAtom.ringConnectivity == -1 && n == 0 || patternAtom.ringConnectivity != -1 && n != patternAtom.ringConnectivity) break;
}}}foundAtom = !foundAtom;
break;
}
return foundAtom;
}, "JS.SmilesAtom,~N");
Clazz.defineMethod (c$, "checkMatchBond", 
 function (patternAtom, atom1, patternBond, iAtom, matchingAtom, bond) {
if (patternBond.bondsOr != null) {
for (var ii = 0; ii < patternBond.nBondsOr; ii++) if (this.checkMatchBond (patternAtom, atom1, patternBond.bondsOr[ii], iAtom, matchingAtom, bond)) return true;

return false;
}if (!this.isRingCheck && !this.isTopology) if (patternBond.primitives == null) {
if (!this.checkPrimitiveBond (patternBond, iAtom, matchingAtom, bond)) return false;
} else {
for (var i = 0; i < patternBond.nPrimitives; i++) if (!this.checkPrimitiveBond (patternBond.primitives[i], iAtom, matchingAtom, bond)) return false;

}patternBond.matchingBond = bond;
return true;
}, "JS.SmilesAtom,JS.SmilesAtom,JS.SmilesBond,~N,~N,JU.Edge");
Clazz.defineMethod (c$, "checkPrimitiveBond", 
 function (patternBond, iAtom1, iAtom2, bond) {
var bondFound = false;
switch (patternBond.order) {
case 96:
return (patternBond.isNot != (this.bioAtoms[iAtom2].getOffsetResidueAtom ("\0", 1) == this.bioAtoms[iAtom1].getOffsetResidueAtom ("\0", 0)));
case 112:
return (patternBond.isNot != this.bioAtoms[iAtom1].isCrossLinked (this.bioAtoms[iAtom2]));
}
var isAromatic1 = (!this.noAromatic && this.bsAromatic.get (iAtom1));
var isAromatic2 = (!this.noAromatic && this.bsAromatic.get (iAtom2));
var order = bond.getCovalentOrder ();
if (isAromatic1 && isAromatic2) {
switch (patternBond.order) {
case 17:
case 65:
bondFound = JS.SmilesSearch.isRingBond (this.ringSets, iAtom1, iAtom2);
break;
case 1:
bondFound = !this.isSmarts || !JS.SmilesSearch.isRingBond (this.ringSets, iAtom1, iAtom2);
break;
case 2:
bondFound = !this.isSmarts || this.aromaticDouble && (order == 2 || order == 514);
break;
case 769:
case 1025:
case 81:
case -1:
bondFound = true;
break;
}
} else {
switch (patternBond.order) {
case 81:
case -1:
bondFound = true;
break;
case 1:
case 257:
case 513:
bondFound = (order == 1 || order == 1041 || order == 1025);
break;
case 769:
bondFound = (order == (this.isSmilesFind ? 33 : 1));
break;
case 1025:
bondFound = (order == (this.isSmilesFind ? 97 : 1));
break;
case 2:
bondFound = (order == 2);
break;
case 3:
bondFound = (order == 3);
break;
case 65:
bondFound = JS.SmilesSearch.isRingBond (this.ringSets, iAtom1, iAtom2);
break;
}
}return bondFound != patternBond.isNot;
}, "JS.SmilesBond,~N,~N,JU.Edge");
c$.isRingBond = Clazz.defineMethod (c$, "isRingBond", 
function (ringSets, i, j) {
return (ringSets != null && ringSets.indexOf ("-" + i + "-" + j + "-") >= 0);
}, "JU.SB,~N,~N");
Clazz.defineMethod (c$, "checkStereochemistry", 
 function () {
for (var i = 0; i < this.measures.size (); i++) if (!this.measures.get (i).check ()) return false;

if (this.stereo != null && !this.stereo.checkStereoChemistry (this, this.v)) return false;
if (this.haveBondStereochemistry) {
for (var k = 0; k < this.ac; k++) {
var sAtom1 = this.patternAtoms[k];
var sAtom2 = null;
var sAtomDirected1 = null;
var sAtomDirected2 = null;
var dir1 = 0;
var dir2 = 0;
var bondType = 0;
var b;
var nBonds = sAtom1.getBondCount ();
var isAtropisomer = false;
for (var j = 0; j < nBonds; j++) {
b = sAtom1.getBond (j);
var isAtom2 = (b.atom2 === sAtom1);
var type = b.order;
switch (type) {
case 769:
case 1025:
case 2:
if (isAtom2) continue;
sAtom2 = b.atom2;
bondType = type;
isAtropisomer = (type != 2);
if (isAtropisomer) dir1 = (b.isNot ? -1 : 1);
break;
case 257:
case 513:
sAtomDirected1 = (isAtom2 ? b.atom1 : b.atom2);
dir1 = (isAtom2 != (type == 257) ? 1 : -1);
break;
}
}
if (isAtropisomer) {
b = sAtom1.getBondNotTo (sAtom2, false);
if (b == null) return false;
sAtomDirected1 = b.getOtherAtom (sAtom1);
b = sAtom2.getBondNotTo (sAtom1, false);
if (b == null) return false;
sAtomDirected2 = b.getOtherAtom (sAtom2);
} else {
if (sAtom2 == null || dir1 == 0) continue;
var a10 = sAtom1;
var nCumulene = 0;
while (sAtom2.getBondCount () == 2 && sAtom2.getValence () == 4) {
nCumulene++;
var e2 = sAtom2.getEdges ();
var e = e2[e2[0].getOtherAtomNode (sAtom2) === a10 ? 1 : 0];
a10 = sAtom2;
sAtom2 = e.getOtherAtomNode (sAtom2);
}
if (nCumulene % 2 == 1) continue;
nBonds = sAtom2.getBondCount ();
for (var j = 0; j < nBonds && dir2 == 0; j++) {
b = sAtom2.getBond (j);
var type = b.order;
switch (type) {
case 257:
case 513:
var isAtom2 = (b.atom2 === sAtom2);
sAtomDirected2 = (isAtom2 ? b.atom1 : b.atom2);
dir2 = (isAtom2 != (type == 257) ? 1 : -1);
break;
}
}
if (dir2 == 0) continue;
}if (this.isSmilesFind) this.setSmilesBondCoordinates (sAtom1, sAtom2, bondType);
var dbAtom1 = sAtom1.getMatchingAtom ();
var dbAtom2 = sAtom2.getMatchingAtom ();
var dbAtom1a = sAtomDirected1.getMatchingAtom ();
var dbAtom2a = sAtomDirected2.getMatchingAtom ();
if (dbAtom1a == null || dbAtom2a == null) return false;
JS.SmilesMeasure.setTorsionData (dbAtom1a, dbAtom1, dbAtom2, dbAtom2a, this.v, isAtropisomer);
if (isAtropisomer) {
dir2 = (bondType == 769 ? 1 : -1);
var f = this.v.vTemp1.dot (this.v.vTemp2);
if (f < 0.05 || f > 0.95 || this.v.vNorm2.dot (this.v.vNorm3) * dir1 * dir2 > 0) return false;
} else {
if (this.v.vTemp1.dot (this.v.vTemp2) * dir1 * dir2 < 0) return false;
}}
}return true;
});
Clazz.defineMethod (c$, "setSmilesBondCoordinates", 
 function (sAtom1, sAtom2, bondType) {
var dbAtom1 = this.jmolAtoms[sAtom1.getMatchingAtomIndex ()];
var dbAtom2 = this.jmolAtoms[sAtom2.getMatchingAtomIndex ()];
dbAtom1.set (-1, 0, 0);
dbAtom2.set (1, 0, 0);
if (bondType == 2) {
var nBonds = 0;
var dir1 = 0;
var bonds = dbAtom1.getEdges ();
for (var k = bonds.length; --k >= 0; ) {
var bond = bonds[k];
if (bond.order == 2) continue;
var atom = bond.getOtherAtomNode (dbAtom1);
atom.set (-1, (nBonds++ == 0) ? -1 : 1, 0);
var mode = (bond.getAtomIndex2 () == dbAtom1.getIndex () ? nBonds : -nBonds);
switch (bond.order) {
case 1025:
dir1 = mode;
break;
case 1041:
dir1 = -mode;
}
}
var dir2 = 0;
nBonds = 0;
var atoms =  new Array (2);
bonds = dbAtom2.getEdges ();
for (var k = bonds.length; --k >= 0; ) {
var bond = bonds[k];
if (bond.order == 2) continue;
var atom = bond.getOtherAtomNode (dbAtom2);
atoms[nBonds] = atom;
atom.set (1, (nBonds++ == 0) ? 1 : -1, 0);
var mode = (bond.getAtomIndex2 () == dbAtom2.getIndex () ? nBonds : -nBonds);
switch (bond.order) {
case 1025:
dir2 = mode;
break;
case 1041:
dir2 = -mode;
}
}
if ((dir1 * dir2 > 0) == (Math.abs (dir1) % 2 == Math.abs (dir2) % 2)) {
var y = (atoms[0]).y;
(atoms[0]).y = (atoms[1]).y;
(atoms[1]).y = y;
}} else {
var bonds = dbAtom1.getEdges ();
var dir = 0;
for (var k = bonds.length; --k >= 0; ) {
var bond = bonds[k];
if (bond.getOtherAtomNode (dbAtom1) === dbAtom2) {
dir = (bond.order == 33 ? 1 : -1);
break;
}}
for (var k = bonds.length; --k >= 0; ) {
var bond = bonds[k];
var atom = bond.getOtherAtomNode (dbAtom1);
if (atom !== dbAtom2) atom.set (-1, 1, 0);
}
bonds = dbAtom2.getEdges ();
for (var k = bonds.length; --k >= 0; ) {
var bond = bonds[k];
var atom = bond.getOtherAtomNode (dbAtom2);
if (atom !== dbAtom1) atom.set (1, 1, -dir / 2.0);
}
}}, "JS.SmilesAtom,JS.SmilesAtom,~N");
Clazz.defineMethod (c$, "getMappedAtoms", 
function (atom, a2, cAtoms) {
var map =  Clazz.newIntArray (cAtoms[4] == null ? 4 : cAtoms[5] == null ? 5 : 6, 0);
for (var i = 0; i < map.length; i++) map[i] = (cAtoms[i] == null ? 104 + i * 100 : cAtoms[i].getIndex ());

var k;
var bonds = atom.getEdges ();
var b2 = (a2 == null ? null : a2.getEdges ());
for (var i = 0; i < map.length; i++) {
for (k = 0; k < bonds.length; k++) if (bonds[k].getOtherAtomNode (atom) === cAtoms[i]) break;

if (k < bonds.length) {
map[i] = (k * 10 + 100) + i;
} else if (a2 != null) {
for (k = 0; k < b2.length; k++) if (b2[k].getOtherAtomNode (a2) === cAtoms[i]) break;

if (k < b2.length) map[i] = (k * 10 + 300) + i;
}}
java.util.Arrays.sort (map);
for (var i = 0; i < map.length; i++) {
map[i] = map[i] % 10;
}
return map;
}, "JU.Node,JU.Node,~A");
Clazz.defineMethod (c$, "createTopoMap", 
function (bsAromatic) {
if (bsAromatic == null) bsAromatic =  new JU.BS ();
var nAtomsMissing = this.getMissingHydrogenCount ();
var atoms =  new Array (this.ac + nAtomsMissing);
this.jmolAtoms = atoms;
var ptAtom = 0;
var bsFixH =  new JU.BS ();
for (var i = 0; i < this.ac; i++) {
var sAtom = this.patternAtoms[i];
var n = sAtom.explicitHydrogenCount;
if (n < 0) n = 0;
var atom = atoms[ptAtom] =  new JS.SmilesAtom ().setAll (0, ptAtom, sAtom.symbol, sAtom.getCharge ());
atom.stereo = sAtom.stereo;
atom.atomName = sAtom.atomName;
atom.residueName = sAtom.residueName;
atom.residueChar = sAtom.residueChar;
atom.residueNumber = sAtom.residueNumber;
atom.atomNumber = sAtom.residueNumber;
atom.explicitHydrogenCount = 0;
atom.isBioAtom = sAtom.isBioAtom;
atom.bioType = sAtom.bioType;
atom.$isLeadAtom = sAtom.$isLeadAtom;
atom.mapIndex = i;
atom.setAtomicMass (sAtom.getAtomicMass ());
if (sAtom.isAromatic ()) bsAromatic.set (ptAtom);
if (!sAtom.isFirst && n == 1 && sAtom.getChiralClass () > 0) bsFixH.set (ptAtom);
sAtom.setMatchingAtom (null, ptAtom++);
var bonds =  new Array (sAtom.getBondCount () + n);
atom.setBonds (bonds);
while (--n >= 0) {
var atomH = atoms[ptAtom] =  new JS.SmilesAtom ().setAll (0, ptAtom, "H", 0);
atomH.mapIndex = -i - 1;
ptAtom++;
atomH.setBonds ( new Array (1));
var b =  new JS.SmilesBond (atom, atomH, 1, false);
if (JU.Logger.debugging) JU.Logger.info ("" + b);
}
}
for (var i = 0; i < this.ac; i++) {
var sAtom = this.patternAtoms[i];
var i1 = sAtom.getMatchingAtomIndex ();
var atom1 = atoms[i1];
var n = sAtom.getBondCount ();
for (var j = 0; j < n; j++) {
var sBond = sAtom.getBond (j);
var firstAtom = (sBond.atom1 === sAtom);
if (firstAtom) {
var order = 1;
switch (sBond.order) {
case 769:
order = 33;
break;
case 1025:
order = 97;
break;
case 257:
order = 1025;
break;
case 513:
order = 1041;
break;
case 112:
case 96:
order = sBond.order;
break;
case 1:
order = 1;
break;
case 17:
order = 514;
break;
case 2:
order = 2;
break;
case 3:
order = 3;
break;
}
var atom2 = atoms[sBond.atom2.getMatchingAtomIndex ()];
var b =  new JS.SmilesBond (atom1, atom2, order, false);
atom2.bondCount--;
if (JU.Logger.debugging) JU.Logger.info ("" + b);
} else {
var atom2 = atoms[sBond.atom1.getMatchingAtomIndex ()];
var b = atom2.getBondTo (atom1);
atom1.addBond (b);
}}
}
for (var i = bsFixH.nextSetBit (0); i >= 0; i = bsFixH.nextSetBit (i + 1)) {
var bonds = atoms[i].getEdges ();
var b = bonds[0];
bonds[0] = bonds[1];
bonds[1] = b;
}
}, "JU.BS");
c$.normalizeAromaticity = Clazz.defineMethod (c$, "normalizeAromaticity", 
function (atoms, bsAromatic, flags) {
var ss =  new JS.SmilesSearch ();
ss.setFlags (flags);
ss.jmolAtoms = atoms;
ss.jmolAtomCount = atoms.length;
ss.bsSelected = JU.BSUtil.newBitSet2 (0, atoms.length);
var vRings = JU.AU.createArrayOfArrayList (4);
ss.setRingData (null, vRings, true);
bsAromatic.or (ss.bsAromatic);
if (!bsAromatic.isEmpty ()) {
var lst = vRings[3];
for (var i = lst.size (); --i >= 0; ) {
var bs = lst.get (i);
for (var j = bs.nextSetBit (0); j >= 0; j = bs.nextSetBit (j + 1)) {
var a = atoms[j];
if (a.isAromatic () || a.elementNumber == -2 || a.elementNumber == 0) continue;
a.setSymbol (a.symbol.toLowerCase ());
}
}
}}, "~A,JU.BS,~N");
Clazz.defineMethod (c$, "setTop", 
function (parent) {
if (parent == null) this.top = this;
 else this.top = parent.getTop ();
}, "JS.SmilesSearch");
Clazz.defineMethod (c$, "getTop", 
function () {
return (this.top === this ? this : this.top.getTop ());
});
Clazz.defineMethod (c$, "getSelections", 
function () {
var ht = this.top.htNested;
if (ht == null || this.jmolAtoms.length == 0) return;
var htNew =  new java.util.Hashtable ();
for (var entry, $entry = ht.entrySet ().iterator (); $entry.hasNext () && ((entry = $entry.next ()) || true);) {
var key = entry.getValue ().toString ();
if (key.startsWith ("select")) {
var bs = (htNew.containsKey (key) ? htNew.get (key) : this.jmolAtoms[0].findAtomsLike (key.substring (6)));
if (bs == null) bs =  new JU.BS ();
htNew.put (key, bs);
entry.setValue (bs);
}}
});
c$.getNormalThroughPoints = Clazz.defineMethod (c$, "getNormalThroughPoints", 
function (pointA, pointB, pointC, vNorm, vAB, vAC) {
vAB.sub2 (pointB, pointA);
vAC.sub2 (pointC, pointA);
vNorm.cross (vAB, vAC);
vNorm.normalize ();
vAB.setT (pointA);
return -vAB.dot (vNorm);
}, "JU.Node,JU.Node,JU.Node,JU.V3,JU.V3,JU.V3");
Clazz.defineStatics (c$,
"NO_AROMATIC", 0x010,
"IGNORE_STEREOCHEMISTRY", 0x020,
"INVERT_STEREOCHEMISTRY", 0x040,
"AROMATIC_DEFINED", 0x080,
"AROMATIC_STRICT", 0x100,
"AROMATIC_DOUBLE", 0x200,
"AROMATIC_MMFF94", 0x700,
"AROMATIC_JSME_NONCANONICAL", 0x800,
"INITIAL_ATOMS", 16);
});
