#/usr/bin/perl

use strict;
use warnings;

use File::Spec;
use List::Util qw(first);

my %ld;
my @results;
my @errors;

die " provide directory path as arg1 perl plotgraph <dir,dir,dir>\n" unless @ARGV > 0;

my $dirs = $ARGV[0];
my @dirs = split/,/,$dirs;

for my $dir ( @dirs ) {

opendir(DIR, $dir) or die $!;

#p-1-n-10000-d-.5.txt

my @files 
        = grep { 
            /^p\-\d+\-n\-\d+\-d\-.\d+.txt/             # p-16-dp-4-fn-300_33_15.txt
	    && -f "$dir/$_"   # and is a file
	} readdir(DIR);

closedir(DIR);

# Loop through the array printing out the filenames

     foreach my $file (@files) {
       
        $file =~ s/\s+// if $file;

        my ($p,$n,$d) = ( $file =~ /^p\-(\d+)\-n\-(\d+)\-d\-(.\d+).txt/ );

        open (FH, File::Spec->catfile($dir,$file)) || die "ERROR Unable to open file: $!\n";

	my $last;
	my $first = <FH>;

	while (<FH>) { $last = $_ }
	close FH;

	# for one line files
        $last=$first unless $last;

        $last =~ s/\s+// if $last;
        chomp $last if $last;
        $last =~ s/\s+// if $last;

        my $msg = '';

        unless ($last and $last =~ /^\d+.\d+$/) {
         $msg = $last if $last;
         $last = -1;
         push @errors,"$file=$p,$n,$d=$last,\"$msg\"";
        }
        else {
         push @results, "$file=$p,$n,$d=$last,\"$msg\"";
        }
        push @{$ld{d}{"d=$d"}{$p}{$n}},$last;
        push @{$ld{p}{"p=$p"}{$n}{$d}},$last;
        push @{$ld{n}{"n=$n"}{$p}{$d}},$last;

    }
}

my $plot = "gplot.txt";
open(FH, '>>', $plot) or die "Failed to open file $plot for writing \n";

for my $res (@results){
   print FH "$res\n";
}
for my $err (@errors){
   print FH "$err\n";
}
close(FH);


# create a speed up hash type too
my %templd;
for my $type ( sort keys %ld ){

   for my $nld ( sort keys %{$ld{$type}} ) {

    for my $ps ( sort keys %{$ld{$type}{$nld}} ) {

      for my $ks ( sort keys %{$ld{$type}{$nld}{$ps}} ) {

         my $speedup=-1;
         my $p1speed=-1;
         my $bestspeed=-1;

         if ( ( $type =~ /^d$/ ) and ( $ld{$type}{$nld}{1}{$ks} ) ){ $p1speed = first { $_ > 0 } sort {$a <=> $b} @{$ld{$type}{$nld}{1}{$ks}}; }
         if ( ( $type =~ /^p$/ ) and ( $ld{$type}{"p=1"}{$ps}{$ks} ) ){ $p1speed = first { $_ > 0 }  sort {$a <=> $b} @{$ld{$type}{"p=1"}{$ps}{$ks}}; }
         if ( ( $type =~ /^n$/ ) and ( $ld{$type}{$nld}{1}{$ks} ) ){ $p1speed = first { $_ > 0 }  sort {$a <=> $b} @{$ld{$type}{$nld}{1}{$ks}}; }

         $bestspeed = first { $_ > 0 } sort {$a <=> $b} @{$ld{$type}{$nld}{$ps}{$ks}};
         $bestspeed=-1 unless $bestspeed;

         push @{$templd{"speedup_$type"}{$nld}{$ps}{$ks}},($p1speed/$bestspeed) if $p1speed > 0 and $bestspeed > 0;

      }
     }
   }
}


for my $type ( sort keys %templd ){
 $ld{$type}=$templd{$type};
}

my %tags = ('d'=>'p','p'=>'n','n'=>'p','speedup_d'=>'p','speedup_p'=>'n','speedup_n'=>'p');

# type = p
for my $type ( sort keys %ld ){

   my @pso;
# nld p=4 etc
   for my $nld ( sort keys %{$ld{$type}} ) {
     push @pso,keys %{$ld{$type}{$nld}};
   }
# all the n values
   my %psoh = map { $_ => 1 } @pso ;

   my $plotnld = "gplot$type.txt";
   open(FH, '>>', $plotnld) or die "Failed to open file $plotnld for writing \n";

       
 for my $nld ( sort keys %{$ld{$type}}){
   
# p,,
  print FH "$type,";
  print FH ",";

# n=100
  for my $p (  sort { $a <=> $b} keys %psoh  ){
   print FH "$tags{$type}=$p,";
  }
  print FH "\n";


   my @kso;
   for my $p ( sort keys %{$ld{$type}{$nld}} ) {
     push @kso,keys %{$ld{$type}{$nld}{$p}};
   }

   my %ksoh = map { $_ => 1 } @kso ;

  for my $k ( sort {$a <=> $b} keys %ksoh ){
      
    print FH "$nld,$k,";
    
    for my $p ( sort { $a <=> $b} keys %psoh ){

     if ( $ld{$type}{$nld}{$p}{$k} and @{$ld{$type}{$nld}{$p}{$k}}){
      # get the best time > 0
      my $tm=-1;
      if ( $type =~ /speedup/ ) {
       $tm = first { $_ > 0 } sort {$b <=> $a} @{$ld{$type}{$nld}{$p}{$k}};
      } else {
       $tm = first { $_ > 0 } sort {$a <=> $b} @{$ld{$type}{$nld}{$p}{$k}};
      }
      $tm='' unless $tm;
      print FH "$tm,";
     }
     else {
       print FH ",";
     }

   }
   print FH "\n";
  }


 }
     close(FH);
}



exit 0;
