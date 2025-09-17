#!/usr/bin/perl -s
#
#  Evgeny Matusov, 2005
#
#
#
#

use Term::ANSIColor qw(:constants);

die "Usage: $0 [options] <input_text>

Normalization script compliant with normalizations at NIST MT evaluations
(assuming a western language as a target language)

Options:
-c        leave case-sensitive
\n\n" if($h||$hh);


while(<>)
{
  print NormalizeText($_);
  print "\n";
}
exit;


sub NormalizeText {
    my ($norm_text) = @_;

# language-independent part:
    $norm_text =~ s/<skipped>//g; # strip "skipped" tags
    $norm_text =~ s/-\n//g; # strip end-of-line hyphenation and join lines
    $norm_text =~ s/\n/ /g; # join lines
#    $norm_text =~ s/(\d)\s+(\d)/$1$2/g; #join digits - THIS IS NOT USED IN mteval-v11b (NIST 2005 evaluation) !
    $norm_text =~ s/&quot;/"/g;  # convert SGML tag for quote to "
    $norm_text =~ s/&amp;/&/g;   # convert SGML tag for ampersand to &
    $norm_text =~ s/&lt;/</g;    # convert SGML tag for less-than to >
    $norm_text =~ s/&gt;/>/g;    # convert SGML tag for greater-than to <

# language-dependent part (assuming Western languages):
    $norm_text = " $norm_text ";
  if(!$c) { $norm_text =~ tr/[A-Z]/[a-z]/; }
    $norm_text =~ s/([\{-\~\[-\` -\&\(-\+\:-\@\/])/ $1 /g;   # tokenize punctuation
    $norm_text =~ s/([^0-9])([\.,])/$1 $2 /g; # tokenize period and comma unless preceded by a digit
    $norm_text =~ s/([\.,])([^0-9])/ $1 $2/g; # tokenize period and comma unless followed by a digit
    $norm_text =~ s/([0-9])(-)/$1 $2 /g; # tokenize dash when preceded by a digit
    $norm_text =~ s/\s+/ /g; # one space only between words
    $norm_text =~ s/^\s+//;  # no leading space
    $norm_text =~ s/\s+$//;  # no trailing space
    return $norm_text;
}

