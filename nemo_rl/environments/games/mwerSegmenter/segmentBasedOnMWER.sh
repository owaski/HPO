#!/bin/bash -e
#
#  Author: Evgeny Matusov, RWTH Aachen
#
##############################################################################

if [ "$1" == "-h" ] ; then
         echo -e "usage:  $0 <source sgm file> <reference sgm file> <translation file> <sysid> <target language> <output sgm file> [normalize]\n"
         echo -e "This script performs segmentation of translation hypotheses based on multiple reference word error rate\n\n"
         echo -e "Required auxiliary scripts: cat sgm2mref.py hyp2sgm.py mWERsegmenter [ normalizeTextNIST.pl ]"
         echo -e "Example: $0 tcstar-run1-epps-test-esen-asr.src.sgm tcstar-run1-epps-test-esen-asr.ref.sgm test.txt RWTH English \\ "
         echo -e "            tcstar-run1-epps-test-esen-asr.RWTH-PRIMARY.sgm normalize\n"
         echo -e "Before using this script, please set the SCRIPTDIR variable to the directory name in which you will store the whole"
         echo -e "package of scripts."
         echo -e "\n"
         exit 0
fi

SCRIPTDIR=`dirname $0`
sourcesgmfile=$1
referencesgmfile=$2
translationfile=$3
sysid=$4
language=$5
outputsgmfile=$6
performNormalization=$7
useCase=$8

tmpdir=`dirname $translationfile`
tmpmref=__mreference
tmpresegmentedfile=__segments
tmptranslationfile=__translation

if [ "$performNormalization" == "normalize" ] ; then
if [ "$useCase" == "1" ] ; then
    echo "$SCRIPTDIR/sgm2mref.py $referencesgmfile | $SCRIPTDIR/normalizeTextNIST.pl -c > $tmpdir/$tmpmref"
    $SCRIPTDIR/sgm2mref.py $referencesgmfile | $SCRIPTDIR/normalizeTextNIST.pl -c > $tmpdir/$tmpmref
    echo "$SCRIPTDIR/normalizeTextNIST.pl -c $translationfile > $tmpdir/$tmptranslationfile"
    $SCRIPTDIR/normalizeTextNIST.pl -c $translationfile > $tmpdir/$tmptranslationfile
else 
    echo "$SCRIPTDIR/sgm2mref.py $referencesgmfile | $SCRIPTDIR/normalizeTextNIST.pl > $tmpdir/$tmpmref"
    $SCRIPTDIR/sgm2mref.py $referencesgmfile | $SCRIPTDIR/normalizeTextNIST.pl > $tmpdir/$tmpmref
    echo "$SCRIPTDIR/normalizeTextNIST.pl $translationfile > $tmpdir/$tmptranslationfile"
    $SCRIPTDIR/normalizeTextNIST.pl $translationfile > $tmpdir/$tmptranslationfile
fi
else
$SCRIPTDIR/sgm2mref.py $referencesgmfile > $tmpdir/$tmpmref
cp $translationfile $tmpdir/$tmptranslationfile
fi

cd $tmpdir
echo "$SCRIPTDIR/mwerSegmenter -mref $tmpmref -hypfile $tmptranslationfile -usecase $useCase"
#$SCRIPTDIR/mwerSegmenter -mref $tmpmref -hypfile $tmptranslationfile -usecase $useCase
$SCRIPTDIR/mwerSegmenter -mref $tmpmref -hypfile $tmptranslationfile -usecase $useCase
cd -
echo "$SCRIPTDIR/hyp2sgm.py --source=$sourcesgmfile --id=$sysid --targetLang=${language} $tmpdir/$tmpresegmentedfile > $outputsgmfile"
$SCRIPTDIR/hyp2sgm.py --source=$sourcesgmfile --id=$sysid --targetLang=${language} $tmpdir/$tmpresegmentedfile > $outputsgmfile

#rm -f $tmpmref $tmpresegmentedfile $tmptranslationfile
