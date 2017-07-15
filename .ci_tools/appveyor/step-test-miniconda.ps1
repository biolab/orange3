
echo "Building and testing using conda in: $env:MINICONDA"
echo ""

if (-not($env:MINICONDA)) { throw "MINICONDA env variable must be defined" }

$python = "$env:MINICONDA\python"
$conda = "$env:MINICONDA\Scripts\conda"

# Need at least conda 4.1.0 (channel priorities)

& "$conda" install --yes "conda>=4.1.0"

# add conda-forge channel
& "$conda" config --append channels conda-forge

# some required packages that are not on conda-forge
& "$conda" config --append channels ales-erjavec

& "$conda" install --yes conda-build=2.1.15

echo "Conda info"
echo "----------"
& "$conda" info
echo ""

echo "Starting conda build"
echo "--------------------"
& "$conda" build conda-recipe

if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }

# also copy build conda pacakge to build artifacts
echo "copying conda package to dist/conda"
echo "-----------------------------------"

$pkgpath = & "$conda" build --output conda-recipe
mkdir -force dist/conda | out-null
cp "$pkgpath" dist/conda/
travis_fold:start:worker_info
[0K[33;1mWorker information[0m
hostname: wjb-2.macstadium-us-se-1.travisci.net:2d0be8bf-6b7d-4da4-ab1f-6b027332435c
version: v2.9.3 https://github.com/travis-ci/worker/tree/a41c772c638071fbbdbc106f31a664c0532e0c36
instance: 80d78bda-ec9a-439c-8ad3-9e09ae8ea728:travis-ci-macos10.12-xcode8.3-1496700148 (via amqp)
startup: 1m35.044616556s
travis_fold:end:worker_info
[0Ktravis_fold:start:system_info
[0K[33;1mBuild system information[0m

Build language: c

Build id: 253524890

Job id: 253524891

travis-build version: 8067d586b

travis_fold:end:system_info
[0K

Fix WWDRCA Certificate

Unable to delete certificate matching "0950B6CD3D2F37EA246A1AAA20DFAADBD6FE1F75"security: AppleWWDRCA.cer: already in /Library/Keychains/System.keychain

$ rvm use

Warning! PATH is not properly set up, '/Users/travis/.rvm/gems/ruby-2.4.1/bin' is not at first place,

         usually this is caused by shell initialization files - check them for 'PATH=...' entries,

         it might also help to re-add RVM to your dotfiles: 'rvm get stable --auto-dotfiles',

         to fix temporarily in this shell session run: 'rvm use ruby-2.4.1'.

[32mUsing /Users/travis/.rvm/gems/ruby-2.4.1[0m

travis_fold:start:git.checkout
[0Ktravis_time:start:226c3b90
[0K$ git clone --depth=50 https://github.com/magnumripper/JohnTheRipper.git magnumripper/JohnTheRipper

Cloning into 'magnumripper/JohnTheRipper'...

remote: Counting objects: 1742, done.[K

remote: Compressing objects:   0% (1/1404)   [K
remote: Compressing objects:   1% (15/1404)   [K
remote: Compressing objects:   2% (29/1404)   [K
remote: Compressing objects:   3% (43/1404)   [K
remote: Compressing objects:   4% (57/1404)   [K
remote: Compressing objects:   5% (71/1404)   [K
remote: Compressing objects:   6% (85/1404)   [K
remote: Compressing objects:   7% (99/1404)   [K
remote: Compressing objects:   8% (113/1404)   [K
remote: Compressing objects:   9% (127/1404)   [K
remote: Compressing objects:  10% (141/1404)   [K
remote: Compressing objects:  11% (155/1404)   [K
remote: Compressing objects:  12% (169/1404)   [K
remote: Compressing objects:  13% (183/1404)   [K
remote: Compressing objects:  14% (197/1404)   [K
remote: Compressing objects:  15% (211/1404)   [K
remote: Compressing objects:  16% (225/1404)   [K
remote: Compressing objects:  17% (239/1404)   [K
remote: Compressing objects:  18% (253/1404)   [K
remote: Compressing objects:  19% (267/1404)   [K
remote: Compressing objects:  20% (281/1404)   [K
remote: Compressing objects:  21% (295/1404)   [K
remote: Compressing objects:  22% (309/1404)   [K
remote: Compressing objects:  23% (323/1404)   [K
remote: Compressing objects:  24% (337/1404)   [K
remote: Compressing objects:  25% (351/1404)   [K
remote: Compressing objects:  26% (366/1404)   [K
remote: Compressing objects:  27% (380/1404)   [K
remote: Compressing objects:  28% (394/1404)   [K
remote: Compressing objects:  29% (408/1404)   [K
remote: Compressing objects:  30% (422/1404)   [K
remote: Compressing objects:  31% (436/1404)   [K
remote: Compressing objects:  32% (450/1404)   [K
remote: Compressing objects:  33% (464/1404)   [K
remote: Compressing objects:  34% (478/1404)   [K
remote: Compressing objects:  35% (492/1404)   [K
remote: Compressing objects:  36% (506/1404)   [K
remote: Compressing objects:  37% (520/1404)   [K
remote: Compressing objects:  38% (534/1404)   [K
remote: Compressing objects:  39% (548/1404)   [K
remote: Compressing objects:  40% (562/1404)   [K
remote: Compressing objects:  41% (576/1404)   [K
remote: Compressing objects:  42% (590/1404)   [K
remote: Compressing objects:  43% (604/1404)   [K
remote: Compressing objects:  44% (618/1404)   [K
remote: Compressing objects:  45% (632/1404)   [K
remote: Compressing objects:  46% (646/1404)   [K
remote: Compressing objects:  47% (660/1404)   [K
remote: Compressing objects:  48% (674/1404)   [K
remote: Compressing objects:  48% (686/1404)   [K
remote: Compressing objects:  49% (688/1404)   [K
remote: Compressing objects:  50% (702/1404)   [K
remote: Compressing objects:  51% (717/1404)   [K
remote: Compressing objects:  52% (731/1404)   [K
remote: Compressing objects:  53% (745/1404)   [K
remote: Compressing objects:  54% (759/1404)   [K
remote: Compressing objects:  55% (773/1404)   [K
remote: Compressing objects:  56% (787/1404)   [K
remote: Compressing objects:  57% (801/1404)   [K
remote: Compressing objects:  58% (815/1404)   [K
remote: Compressing objects:  59% (829/1404)   [K
remote: Compressing objects:  60% (843/1404)   [K
remote: Compressing objects:  61% (857/1404)   [K
remote: Compressing objects:  62% (871/1404)   [K
remote: Compressing objects:  63% (885/1404)   [K
remote: Compressing objects:  64% (899/1404)   [K
remote: Compressing objects:  65% (913/1404)   [K
remote: Compressing objects:  66% (927/1404)   [K
remote: Compressing objects:  67% (941/1404)   [K
remote: Compressing objects:  68% (955/1404)   [K
remote: Compressing objects:  69% (969/1404)   [K
remote: Compressing objects:  70% (983/1404)   [K
remote: Compressing objects:  71% (997/1404)   [K
remote: Compressing objects:  72% (1011/1404)   [K
remote: Compressing objects:  73% (1025/1404)   [K
remote: Compressing objects:  74% (1039/1404)   [K
remote: Compressing objects:  75% (1053/1404)   [K
remote: Compressing objects:  76% (1068/1404)   [K
remote: Compressing objects:  77% (1082/1404)   [K
remote: Compressing objects:  78% (1096/1404)   [K
remote: Compressing objects:  79% (1110/1404)   [K
remote: Compressing objects:  80% (1124/1404)   [K
remote: Compressing objects:  81% (1138/1404)   [K
remote: Compressing objects:  82% (1152/1404)   [K
remote: Compressing objects:  83% (1166/1404)   [K
remote: Compressing objects:  84% (1180/1404)   [K
remote: Compressing objects:  85% (1194/1404)   [K
remote: Compressing objects:  86% (1208/1404)   [K
remote: Compressing objects:  87% (1222/1404)   [K
remote: Compressing objects:  88% (1236/1404)   [K
remote: Compressing objects:  89% (1250/1404)   [K
remote: Compressing objects:  90% (1264/1404)   [K
remote: Compressing objects:  91% (1278/1404)   [K
remote: Compressing objects:  92% (1292/1404)   [K
remote: Compressing objects:  93% (1306/1404)   [K
remote: Compressing objects:  94% (1320/1404)   [K
remote: Compressing objects:  95% (1334/1404)   [K
remote: Compressing objects:  96% (1348/1404)   [K
remote: Compressing objects:  97% (1362/1404)   [K
remote: Compressing objects:  98% (1376/1404)   [K
remote: Compressing objects:  99% (1390/1404)   [K
remote: Compressing objects: 100% (1404/1404)   [K
remote: Compressing objects: 100% (1404/1404), done.[K

Receiving objects:   0% (1/1742)   
Receiving objects:   1% (18/1742)   
Receiving objects:   2% (35/1742)   
Receiving objects:   3% (53/1742)   
Receiving objects:   4% (70/1742)   
Receiving objects:   5% (88/1742)   
Receiving objects:   6% (105/1742)   
Receiving objects:   7% (122/1742)   
Receiving objects:   8% (140/1742)   
Receiving objects:   9% (157/1742)   
Receiving objects:  10% (175/1742)   
Receiving objects:  10% (183/1742), 5.97 MiB | 5.95 MiB/s   
Receiving objects:  11% (192/1742), 5.97 MiB | 5.95 MiB/s   
Receiving objects:  12% (210/1742), 5.97 MiB | 5.95 MiB/s   
Receiving objects:  13% (227/1742), 5.97 MiB | 5.95 MiB/s   
Receiving objects:  14% (244/1742), 5.97 MiB | 5.95 MiB/s   
Receiving objects:  15% (262/1742), 10.91 MiB | 7.26 MiB/s   
Receiving objects:  15% (273/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  16% (279/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  17% (297/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  18% (314/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  19% (331/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  20% (349/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  21% (366/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  22% (384/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  23% (401/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  24% (419/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  25% (436/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  26% (453/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  27% (471/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  28% (488/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  29% (506/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  30% (523/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  31% (541/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  32% (558/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  33% (575/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  34% (593/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  35% (610/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  36% (628/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  37% (645/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  38% (662/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  39% (680/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  40% (697/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  41% (715/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  42% (732/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  43% (750/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  44% (767/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  45% (784/1742), 16.01 MiB | 7.97 MiB/s   
Receiving objects:  46% (802/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  47% (819/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  48% (837/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  49% (854/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  50% (871/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  51% (889/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  52% (906/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  53% (924/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  54% (941/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  55% (959/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  56% (976/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  57% (993/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  58% (1011/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  59% (1028/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  60% (1046/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  61% (1063/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  62% (1081/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  63% (1098/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  64% (1115/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  65% (1133/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  66% (1150/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  67% (1168/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  68% (1185/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  69% (1202/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  70% (1220/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  71% (1237/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  72% (1255/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  73% (1272/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  74% (1290/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  75% (1307/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  76% (1324/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  77% (1342/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  78% (1359/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  79% (1377/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  80% (1394/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  81% (1412/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  82% (1429/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  83% (1446/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  84% (1464/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  85% (1481/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  86% (1499/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  87% (1516/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  88% (1533/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  89% (1551/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  90% (1568/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  91% (1586/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  92% (1603/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  93% (1621/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  94% (1638/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  95% (1655/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  96% (1673/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  97% (1690/1742), 21.86 MiB | 8.71 MiB/s   
remote: Total 1742 (delta 493), reused 752 (delta 331), pack-reused 0[K

Receiving objects:  98% (1708/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects:  99% (1725/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects: 100% (1742/1742), 21.86 MiB | 8.71 MiB/s   
Receiving objects: 100% (1742/1742), 25.01 MiB | 8.71 MiB/s, done.

Resolving deltas:   0% (0/493)   
Resolving deltas:   1% (7/493)   
Resolving deltas:   2% (10/493)   
Resolving deltas:   3% (15/493)   
Resolving deltas:   4% (20/493)   
Resolving deltas:   5% (25/493)   
Resolving deltas:   6% (30/493)   
Resolving deltas:   7% (35/493)   
Resolving deltas:   8% (40/493)   
Resolving deltas:   9% (45/493)   
Resolving deltas:  10% (50/493)   
Resolving deltas:  14% (72/493)   
Resolving deltas:  16% (80/493)   
Resolving deltas:  17% (86/493)   
Resolving deltas:  18% (89/493)   
Resolving deltas:  19% (94/493)   
Resolving deltas:  20% (99/493)   
Resolving deltas:  31% (157/493)   
Resolving deltas:  32% (161/493)   
Resolving deltas:  33% (165/493)   
Resolving deltas:  34% (169/493)   
Resolving deltas:  35% (174/493)   
Resolving deltas:  37% (184/493)   
Resolving deltas:  38% (188/493)   
Resolving deltas:  39% (193/493)   
Resolving deltas:  40% (199/493)   
Resolving deltas:  41% (203/493)   
Resolving deltas:  43% (215/493)   
Resolving deltas:  44% (217/493)   
Resolving deltas:  45% (223/493)   
Resolving deltas:  46% (229/493)   
Resolving deltas:  47% (232/493)   
Resolving deltas:  48% (240/493)   
Resolving deltas:  49% (242/493)   
Resolving deltas:  50% (247/493)   
Resolving deltas:  51% (252/493)   
Resolving deltas:  52% (259/493)   
Resolving deltas:  53% (262/493)   
Resolving deltas:  54% (268/493)   
Resolving deltas:  55% (272/493)   
Resolving deltas:  56% (279/493)   
Resolving deltas:  57% (284/493)   
Resolving deltas:  58% (290/493)   
Resolving deltas:  59% (291/493)   
Resolving deltas:  60% (297/493)   
Resolving deltas:  61% (301/493)   
Resolving deltas:  62% (307/493)   
Resolving deltas:  63% (312/493)   
Resolving deltas:  65% (321/493)   
Resolving deltas:  66% (328/493)   
Resolving deltas:  67% (332/493)   
Resolving deltas:  68% (336/493)   
Resolving deltas:  69% (341/493)   
Resolving deltas:  70% (350/493)   
Resolving deltas:  71% (351/493)   
Resolving deltas:  72% (355/493)   
Resolving deltas:  73% (361/493)   
Resolving deltas:  74% (365/493)   
Resolving deltas:  75% (370/493)   
Resolving deltas:  76% (377/493)   
Resolving deltas:  77% (380/493)   
Resolving deltas:  78% (385/493)   
Resolving deltas:  79% (390/493)   
Resolving deltas:  80% (396/493)   
Resolving deltas:  81% (401/493)   
Resolving deltas:  82% (405/493)   
Resolving deltas:  83% (412/493)   
Resolving deltas:  84% (415/493)   
Resolving deltas:  85% (421/493)   
Resolving deltas:  86% (426/493)   
Resolving deltas:  87% (429/493)   
Resolving deltas:  88% (434/493)   
Resolving deltas:  89% (439/493)   
Resolving deltas:  90% (444/493)   
Resolving deltas:  91% (449/493)   
Resolving deltas:  92% (454/493)   
Resolving deltas:  93% (459/493)   
Resolving deltas:  94% (464/493)   
Resolving deltas:  95% (469/493)   
Resolving deltas:  96% (475/493)   
Resolving deltas:  97% (479/493)   
Resolving deltas:  98% (484/493)   
Resolving deltas:  99% (489/493)   
Resolving deltas: 100% (493/493)   
Resolving deltas: 100% (493/493), done.

Checking out files:  18% (239/1327)   
Checking out files:  19% (253/1327)   
Checking out files:  20% (266/1327)   
Checking out files:  21% (279/1327)   
Checking out files:  22% (292/1327)   
Checking out files:  23% (306/1327)   
Checking out files:  24% (319/1327)   
Checking out files:  25% (332/1327)   
Checking out files:  26% (346/1327)   
Checking out files:  27% (359/1327)   
Checking out files:  28% (372/1327)   
Checking out files:  29% (385/1327)   
Checking out files:  30% (399/1327)   
Checking out files:  31% (412/1327)   
Checking out files:  32% (425/1327)   
Checking out files:  33% (438/1327)   
Checking out files:  34% (452/1327)   
Checking out files:  35% (465/1327)   
Checking out files:  36% (478/1327)   
Checking out files:  37% (491/1327)   
Checking out files:  38% (505/1327)   
Checking out files:  39% (518/1327)   
Checking out files:  40% (531/1327)   
Checking out files:  41% (545/1327)   
Checking out files:  42% (558/1327)   
Checking out files:  43% (571/1327)   
Checking out files:  44% (584/1327)   
Checking out files:  45% (598/1327)   
Checking out files:  46% (611/1327)   
Checking out files:  47% (624/1327)   
Checking out files:  48% (637/1327)   
Checking out files:  49% (651/1327)   
Checking out files:  50% (664/1327)   
Checking out files:  51% (677/1327)   
Checking out files:  52% (691/1327)   
Checking out files:  53% (704/1327)   
Checking out files:  54% (717/1327)   
Checking out files:  55% (730/1327)   
Checking out files:  56% (744/1327)   
Checking out files:  57% (757/1327)   
Checking out files:  58% (770/1327)   
Checking out files:  59% (783/1327)   
Checking out files:  60% (797/1327)   
Checking out files:  61% (810/1327)   
Checking out files:  62% (823/1327)   
Checking out files:  63% (837/1327)   
Checking out files:  64% (850/1327)   
Checking out files:  65% (863/1327)   
Checking out files:  66% (876/1327)   
Checking out files:  67% (890/1327)   
Checking out files:  68% (903/1327)   
Checking out files:  69% (916/1327)   
Checking out files:  70% (929/1327)   
Checking out files:  71% (943/1327)   
Checking out files:  72% (956/1327)   
Checking out files:  73% (969/1327)   
Checking out files:  74% (982/1327)   
Checking out files:  75% (996/1327)   
Checking out files:  76% (1009/1327)   
Checking out files:  77% (1022/1327)   
Checking out files:  78% (1036/1327)   
Checking out files:  79% (1049/1327)   
Checking out files:  80% (1062/1327)   
Checking out files:  81% (1075/1327)   
Checking out files:  82% (1089/1327)   
Checking out files:  83% (1102/1327)   
Checking out files:  84% (1115/1327)   
Checking out files:  85% (1128/1327)   
Checking out files:  86% (1142/1327)   
Checking out files:  87% (1155/1327)   
Checking out files:  88% (1168/1327)   
Checking out files:  89% (1182/1327)   
Checking out files:  90% (1195/1327)   
Checking out files:  91% (1208/1327)   
Checking out files:  92% (1221/1327)   
Checking out files:  93% (1235/1327)   
Checking out files:  94% (1248/1327)   
Checking out files:  95% (1261/1327)   
Checking out files:  96% (1274/1327)   
Checking out files:  97% (1288/1327)   
Checking out files:  98% (1301/1327)   
Checking out files:  99% (1314/1327)   
Checking out files: 100% (1327/1327)   
Checking out files: 100% (1327/1327), done.



travis_time:end:226c3b90:start=1500023595455298000,finish=1500023602899184000,duration=7443886000
[0K$ cd magnumripper/JohnTheRipper

travis_time:start:05725b7f
[0K$ git fetch origin +refs/pull/2623/merge:

remote: Counting objects: 27, done.[K

remote: Compressing objects:  20% (1/5)   [K
remote: Compressing objects:  40% (2/5)   [K
remote: Compressing objects:  60% (3/5)   [K
remote: Compressing objects:  80% (4/5)   [K
remote: Compressing objects: 100% (5/5)   [K
remote: Compressing objects: 100% (5/5), done.[K

remote: Total 27 (delta 23), reused 25 (delta 22), pack-reused 0[K

Unpacking objects:   3% (1/27)   
Unpacking objects:   7% (2/27)   
Unpacking objects:  11% (3/27)   
Unpacking objects:  14% (4/27)   
Unpacking objects:  18% (5/27)   
Unpacking objects:  22% (6/27)   
Unpacking objects:  25% (7/27)   
Unpacking objects:  29% (8/27)   
Unpacking objects:  33% (9/27)   
Unpacking objects:  37% (10/27)   
Unpacking objects:  40% (11/27)   
Unpacking objects:  44% (12/27)   
Unpacking objects:  48% (13/27)   
Unpacking objects:  51% (14/27)   
Unpacking objects:  55% (15/27)   
Unpacking objects:  59% (16/27)   
Unpacking objects:  62% (17/27)   
Unpacking objects:  66% (18/27)   
Unpacking objects:  70% (19/27)   
Unpacking objects:  74% (20/27)   
Unpacking objects:  77% (21/27)   
Unpacking objects:  81% (22/27)   
Unpacking objects:  85% (23/27)   
Unpacking objects:  88% (24/27)   
Unpacking objects:  92% (25/27)   
Unpacking objects:  96% (26/27)   
Unpacking objects: 100% (27/27)   
Unpacking objects: 100% (27/27), done.

From https://github.com/magnumripper/JohnTheRipper

 * branch            refs/pull/2623/merge -> FETCH_HEAD



travis_time:end:05725b7f:start=1500023602967272000,finish=1500023603430331000,duration=463059000
[0K$ git checkout -qf FETCH_HEAD

travis_fold:end:git.checkout
[0Ktravis_fold:start:git.submodule
[0Ktravis_time:start:1b230bbc
[0K$ git submodule update --init --recursive



travis_time:end:1b230bbc:start=1500023603595138000,finish=1500023603834926000,duration=239788000
[0Ktravis_fold:end:git.submodule
[0Ktravis_fold:start:services
[0Ktravis_time:start:38d654c2
[0K$ sudo service docker start

sudo: service: command not found



travis_time:end:38d654c2:start=1500023603867827000,finish=1500023603988703000,duration=120876000
[0Ktravis_fold:end:services
[0K

[33;1mSetting environment variables from .travis.yml[0m

$ export ASAN=""

$ export OPENCL="yes"



$ export CC=gcc

$ gcc --version

Configured with: --prefix=/Applications/Xcode.app/Contents/Developer/usr --with-gxx-include-dir=/usr/include/c++/4.2.1

Apple LLVM version 8.1.0 (clang-802.0.42)

Target: x86_64-apple-darwin16.6.0

Thread model: posix

InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

travis_fold:start:before_install
[0Ktravis_time:start:00345051
[0K$ export OMP_NUM_THREADS=4



travis_time:end:00345051:start=1500023609210155000,finish=1500023609230240000,duration=20085000
[0Ktravis_fold:end:before_install
[0Ktravis_time:start:133983c1
[0K$ .travis/check.sh

checking build system type... x86_64-apple-darwin16.6.0

checking host system type... x86_64-apple-darwin16.6.0

checking whether to compile using MPI... no

checking for gcc... gcc

checking whether the C compiler works... yes

checking for C compiler default output file name... a.out

checking for suffix of executables... 

checking whether we are cross compiling... no

checking for suffix of object files... o

checking whether we are using the GNU C compiler... yes

checking whether gcc accepts -g... yes

checking for gcc option to accept ISO C89... none needed

checking whether gcc understands -c and -o together... yes

checking whether we are using the GNU C compiler... (cached) yes

checking whether gcc accepts -g... (cached) yes

checking for gcc option to accept ISO C89... (cached) none needed

checking whether gcc understands -c and -o together... (cached) yes

checking additional paths...  -L/usr/local/lib -I/usr/local/include

checking arg check macro for -m with gcc... yes

checking arg check macro for -Q with gcc... yes

checking if gcc supports -funroll-loops... yes

checking if gcc supports -Os... yes

checking if gcc supports -finline-functions... yes

checking if gcc supports -Og... no

checking if gcc supports -Wall... yes

checking if gcc supports -Wdeclaration-after-statement... yes

checking if gcc supports -fomit-frame-pointer... yes

checking if gcc supports --param allow-store-data-races=0... no

checking if gcc supports -Wno-deprecated-declarations... yes

checking if gcc supports -Wformat-extra-args... yes

checking if gcc supports  -Wunused-but-set-variable... no

checking if gcc supports -Qunused-arguments... yes

checking if gcc supports -std=gnu89... yes

checking if gcc supports -Wdate-time... yes

checking whether ln -s works... yes

checking for grep that handles long lines and -e... /usr/bin/grep

checking for a sed that does not truncate output... /usr/bin/sed

checking for GNU make... make

checking whether make sets $(MAKE)... yes

checking how to run the C preprocessor... gcc -E

checking for a thread-safe mkdir -p... /usr/local/bin/gmkdir -p

checking for sort... /usr/bin/sort

checking for find... /usr/bin/find

checking for perl... /usr/bin/perl

checking for ar... ar

checking for strip... strip

checking for pkg-config... /usr/local/bin/pkg-config

checking pkg-config is at least version 0.9.0... yes

checking if pkg-config will be used... yes

checking for egrep... /usr/bin/grep -E

checking for ANSI C header files... yes

checking for sys/types.h... yes

checking for sys/stat.h... yes

checking for stdlib.h... yes

checking for string.h... yes

checking for memory.h... yes

checking for strings.h... yes

checking for inttypes.h... yes

checking for stdint.h... yes

checking for unistd.h... yes

checking size of short... 2

checking size of int... 4

checking size of long... 8

checking size of long long... 8

checking size of wchar_t... 4

checking size of int *... 8

checking size of void *... 8

checking whether OS X 'as' needs -q option... no

configure: Testing build host's native CPU features

checking for Hyperthreading... no

checking for MMX... yes

checking for SSE2... yes

checking for SSSE3... yes

checking for SSE4.1... yes

checking for AVX... no

checking for arch.h alternative... x86-64.h

checking whether compiler understands -march=native... yes

checking for 32/64 bit... 64-bit

checking for extra ASFLAGS...  -DUNDERSCORES -DBSD -DALIGN_LOG

checking for X32 ABI... no

checking for byte ordering according to target triple... little

checking for OPENSSL... yes

checking for sqrt in -lm... yes

checking for deflate in -lz... yes

checking for library containing crypt... none required

checking gmp.h usability... yes

checking gmp.h presence... yes

checking for gmp.h... yes

checking for __gmpz_init in -lgmp... yes

checking skey.h usability... no

checking skey.h presence... no

checking for skey.h... no

checking for S/Key... using our own code

checking bzlib.h usability... yes

checking bzlib.h presence... yes

checking for bzlib.h... yes

checking for main in -lbz2... yes

checking for main in -lkernel32... no

checking for dlopen in -ldl... yes

checking intrin.h usability... no

checking intrin.h presence... no

checking for intrin.h... no

checking librexgen/version.h usability... no

checking librexgen/version.h presence... no

checking for librexgen/version.h... no

checking pcap.h usability... yes

checking pcap.h presence... yes

checking for pcap.h... yes

checking for pcap_compile in -lpcap... yes

checking for pcap.h... (cached) yes

checking for pcap_compile in -lwpcap... no

checking whether time.h and sys/time.h may both be included... yes

checking whether string.h and strings.h may both be included... yes

checking for SHA256... yes

checking for WHIRLPOOL... yes

checking for RIPEMD160... yes

checking for AES_encrypt... yes

checking for gcc option to support OpenMP... unsupported

checking additional paths for OpenCL... none

checking if compiler needs -Werror to reject unknown flags... yes

checking whether pthreads work with -pthread... yes

checking for joinable pthread attribute... PTHREAD_CREATE_JOINABLE

checking if more special flags are required for pthreads... -D_THREAD_SAFE

checking for PTHREAD_PRIO_INHERIT... yes

checking whether we are using the Microsoft C compiler... no

checking CL/cl.h usability... no

checking CL/cl.h presence... no

checking for CL/cl.h... no

checking OpenCL/cl.h usability... yes

checking OpenCL/cl.h presence... yes

checking for OpenCL/cl.h... yes

checking windows.h usability... no

checking windows.h presence... no

checking for windows.h... no

checking for OpenCL library... -Wl,-framework,OpenCL

checking arpa/inet.h usability... yes

checking arpa/inet.h presence... yes

checking for arpa/inet.h... yes

checking crypt.h usability... no

checking crypt.h presence... no

checking for crypt.h... no

checking dirent.h usability... yes

checking dirent.h presence... yes

checking for dirent.h... yes

checking fcntl.h usability... yes

checking fcntl.h presence... yes

checking for fcntl.h... yes

checking limits.h usability... yes

checking limits.h presence... yes

checking for limits.h... yes

checking locale.h usability... yes

checking locale.h presence... yes

checking for locale.h... yes

checking malloc.h usability... no

checking malloc.h presence... no

checking for malloc.h... no

checking net/ethernet.h usability... yes

checking net/ethernet.h presence... yes

checking for net/ethernet.h... yes

checking netdb.h usability... yes

checking netdb.h presence... yes

checking for netdb.h... yes

checking netinet/in.h usability... yes

checking netinet/in.h presence... yes

checking for netinet/in.h... yes

checking netinet/in_systm.h usability... yes

checking netinet/in_systm.h presence... yes

checking for netinet/in_systm.h... yes

checking for string.h... (cached) yes

checking for strings.h... (cached) yes

checking sys/ethernet.h usability... no

checking sys/ethernet.h presence... no

checking for sys/ethernet.h... no

checking sys/file.h usability... yes

checking sys/file.h presence... yes

checking for sys/file.h... yes

checking sys/param.h usability... yes

checking sys/param.h presence... yes

checking for sys/param.h... yes

checking sys/socket.h usability... yes

checking sys/socket.h presence... yes

checking for sys/socket.h... yes

checking sys/time.h usability... yes

checking sys/time.h presence... yes

checking for sys/time.h... yes

checking sys/times.h usability... yes

checking sys/times.h presence... yes

checking for sys/times.h... yes

checking for sys/types.h... (cached) yes

checking termios.h usability... yes

checking termios.h presence... yes

checking for termios.h... yes

checking for unistd.h... (cached) yes

checking unixlib/local.h usability... no

checking unixlib/local.h presence... no

checking for unixlib/local.h... no

checking for windows.h... (cached) no

checking for net/if.h... yes

checking for net/if_arp.h... yes

checking for netinet/if_ether.h... yes

checking for netinet/ip.h... yes

checking for stdbool.h that conforms to C99... yes

checking for _Bool... yes

checking for inline... inline

checking for int32_t... yes

checking for int64_t... yes

checking for off_t... yes

checking for size_t... yes

checking for ssize_t... yes

checking for uint16_t... yes

checking for uint32_t... yes

checking for uint64_t... yes

checking for uint8_t... yes

checking for ptrdiff_t... yes

checking for int128... no

checking for __int128... yes

checking for __int128_t... yes

checking for error_at_line... no

checking for pid_t... yes

checking vfork.h usability... no

checking vfork.h presence... no

checking for vfork.h... no

checking for fork... yes

checking for vfork... yes

checking for working fork... yes

checking for working vfork... (cached) yes

checking for fseek64... no

checking for fseeko... yes

checking for fseeko64... no

checking for _fseeki64... no

checking for lseek64... no

checking for lseek... yes

checking for ftell64... no

checking for ftello... yes

checking for ftello64... no

checking for _ftelli64... no

checking for fopen64... no

checking for _fopen64... no

checking for memmem... yes

checking for mmap... yes

checking for sleep... yes

checking for setenv... yes

checking for putenv... yes

checking for strcasecmp... yes

checking for strncasecmp... yes

checking for stricmp... no

checking for strcmpi... no

checking for _stricmp... no

checking for _strcmpi... no

checking for strnicmp... no

checking for strncmpi... no

checking for _strnicmp... no

checking for _strncmpi... no

checking for strnlen... yes

checking for strlwr... no

checking for strupr... no

checking for strrev... no

checking for atoll... yes

checking for _atoi64... no

checking for snprintf... yes

checking for sprintf_s... no

checking for strcasestr... yes

checking for clGetKernelArgInfo... yes

checking for posix_memalign... yes

checking for yasm that supports "--prefix=_ -f macho64"... 

checking for OS-specific feature macros needed... none

checking size of size_t... 8

checking size of off_t... 8

configure: Fuzz check disabled

configure: Fuzzing (using libFuzzer) check disabled

configure: creating *_plug.c and OpenCL object rules

configure: creating Makefile dependencies

configure: creating ./john_build_rule.h

configure: creating ./config.status

config.status: creating Makefile

config.status: creating aes/Makefile

config.status: creating aes/aesni/Makefile

config.status: creating aes/openssl/Makefile

config.status: creating secp256k1/Makefile

config.status: creating escrypt/Makefile

config.status: creating autoconfig.h

config.status: linking x86-64.h to arch.h

config.status: executing default commands

configure: creating ./fmt_externs.h

configure: creating ./fmt_registers.h



Configured for building John the Ripper jumbo:



Target CPU ................................. x86_64 SSE4.1, 64-bit LE

AES-NI support ............................. depends on OpenSSL

Target OS .................................. darwin16.6.0

Cross compiling ............................ no

Legacy arch header ......................... x86-64.h



Optional libraries/features found:

Fuzzing test ............................... no

Experimental code .......................... no

OpenMPI support (default disabled) ......... no

Fork support ............................... yes

OpenMP support ............................. no

OpenCL support ............................. yes

Generic crypt(3) format .................... yes

Rexgen (extra cracking mode) ............... no

GMP (PRINCE mode and faster SRP formats) ... yes

PCAP (vncpcap2john and SIPdump) ............ yes

Z (pkzip format, gpg2john) ................. yes

BZ2 (gpg2john extra decompression logic) ... yes

128-bit integer (faster PRINCE mode) ....... yes

Memory map (share/page large files) ........ yes

ZTEX USB-FPGA module 1.15y support ......... no



Development options (these may hurt performance when enabled):

Memdbg memory debugging settings ........... disabled

AddressSanitizer ("ASan") .................. disabled

UndefinedBehaviorSanitizer ("UbSan") ....... disabled



Install missing libraries to get any needed features that were omitted.



Configure finished.  Now 'make clean && make -s' to compile.

ar: creating archive aes.a

ar: creating archive secp256k1.a

[1mssh_ng_fmt_plug.c:312:19: [0m[0;1;35mwarning: [0m[1munused function 'check_padding_only' [-Wunused-function][0m

inline static int check_padding_only(unsigned char *out, int length)

[0;1;32m                  ^

[0m1 warning generated.

clang: [0;1;35mwarning: [0margument unused during compilation: '-pthread' [-Wunused-command-line-argument][0m

clang: [0;1;35mwarning: [0margument unused during compilation: '-pthread' [-Wunused-command-line-argument][0m

clang: [0;1;35mwarning: [0margument unused during compilation: '-pthread' [-Wunused-command-line-argument][0m

clang: [0;1;35mwarning: [0margument unused during compilation: '-pthread' [-Wunused-command-line-argument][0m



Make process completed.

Testing: descrypt, traditional crypt(3) [DES 128/128 SSE2-16]... /-\|/-PASS

Testing: bsdicrypt, BSDI crypt(3) ("_J9..", 725 iterations) [DES 128/128 SSE2-16]... \|/-\|/-\|/-\|PASS

Testing: md5crypt, crypt(3) $1$ [MD5 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: bcrypt ("$2a$05", 32 iterations) [Blowfish 32/64 X3]... |/-\|/-\|/-\|/-PASS

Testing: scrypt (16384, 8, 1) [Salsa20/8 128/128 SSE2]... \|/-\|/-\|/PASS

Testing: LM [DES 128/128 SSE2-16]... -\|/-\|/-\|PASS

Testing: AFS, Kerberos AFS [DES 48/64 4K]... /-\|/-\|/PASS

Testing: tripcode [DES 128/128 SSE2-16]... -\|/PASS

Testing: dynamic_0 [md5($p) (raw-md5) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1 [md5($p.$s) (joomla) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_2 [md5(md5($p)) (e107) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_3 [md5(md5(md5($p))) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_4 [md5($s.$p) (OSC) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_5 [md5($s.$p.$s) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_6 [md5(md5($p).$s) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_8 [md5(md5($s).$p) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_9 [md5($s.md5($p)) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_10 [md5($s.md5($s.$p)) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_11 [md5($s.md5($p.$s)) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_12 [md5(md5($s).md5($p)) (IPB) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_13 [md5(md5($p).md5($s)) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_14 [md5($s.md5($p).$s) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_15 [md5($u.md5($p).$s) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_16 [md5(md5(md5($p).$s).$s2) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_18 [md5($s.Y.$p.0xF7.$s) (Post.Office MD5) 32/64 x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_19 [md5($p) (Cisco PIX) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_20 [md5($p.$s) (Cisco ASA) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_22 [md5(sha1($p)) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_23 [sha1(md5($p)) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_24 [sha1($p.$s) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_25 [sha1($s.$p) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_26 [sha1($p) raw-sha1 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_29 [md5(utf16($p)) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_30 [md4($p) (raw-md4) 128/128 SSE4.1 4x4]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_31 [md4($s.$p) 128/128 SSE4.1 4x4]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_32 [md4($p.$s) 128/128 SSE4.1 4x4]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_33 [md4(utf16($p)) 128/128 SSE4.1 4x4]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_34 [md5(md4($p)) 128/128 SSE4.1 4x4]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_35 [sha1(uc($u).:.$p) (ManGOS) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_36 [sha1($u.:.$p) (ManGOS2) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_37 [sha1(lc($u).$p) (SMF) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_38 [sha1($s.sha1($s.sha1($p))) (Wolt3BB) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_39 [md5($s.pad16($p)) (net-md5) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_40 [sha1($s.pad20($p)) (net-sha1) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_50 [sha224($p) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_51 [sha224($s.$p) 128/128 SSE4.1 4x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_52 [sha224($p.$s) 128/128 SSE4.1 4x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_53 [sha224(sha224($p)) 128/128 SSE4.1 4x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_54 [sha224(sha224_raw($p)) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_55 [sha224(sha224($p).$s) 128/128 SSE4.1 4x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_56 [sha224($s.sha224($p)) 128/128 SSE4.1 4x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_57 [sha224(sha224($s).sha224($p)) 128/128 SSE4.1 4x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_58 [sha224(sha224($p).sha224($p)) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_60 [sha256($p) 128/128 SSE4.1 4x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_61 [sha256($s.$p) 128/128 SSE4.1 4x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_62 [sha256($p.$s) 128/128 SSE4.1 4x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_63 [sha256(sha256($p)) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_64 [sha256(sha256_raw($p)) 128/128 SSE4.1 4x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_65 [sha256(sha256($p).$s) 128/128 SSE4.1 4x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_66 [sha256($s.sha256($p)) 128/128 SSE4.1 4x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_67 [sha256(sha256($s).sha256($p)) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_68 [sha256(sha256($p).sha256($p)) 128/128 SSE4.1 4x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_70 [sha384($p) 128/128 SSE4.1 2x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_71 [sha384($s.$p) 128/128 SSE4.1 2x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_72 [sha384($p.$s) 128/128 SSE4.1 2x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_73 [sha384(sha384($p)) 128/128 SSE4.1 2x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_74 [sha384(sha384_raw($p)) 128/128 SSE4.1 2x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_75 [sha384(sha384($p).$s) 128/128 SSE4.1 2x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_76 [sha384($s.sha384($p)) 128/128 SSE4.1 2x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_77 [sha384(sha384($s).sha384($p)) 128/128 SSE4.1 2x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_78 [sha384(sha384($p).sha384($p)) 128/128 SSE4.1 2x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_80 [sha512($p) 128/128 SSE4.1 2x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_81 [sha512($s.$p) 128/128 SSE4.1 2x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_82 [sha512($p.$s) 128/128 SSE4.1 2x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_83 [sha512(sha512($p)) 128/128 SSE4.1 2x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_84 [sha512(sha512_raw($p)) 128/128 SSE4.1 2x]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_85 [sha512(sha512($p).$s) 128/128 SSE4.1 2x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_86 [sha512($s.sha512($p)) 128/128 SSE4.1 2x]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_90 [gost($p) 64/64]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_91 [gost($s.$p) 64/64]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_92 [gost($p.$s) 64/64]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_93 [gost(gost($p)) 64/64]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_94 [gost(gost_raw($p)) 64/64]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_95 [gost(gost($p).$s) 64/64]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_96 [gost($s.gost($p)) 64/64]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_97 [gost(gost($s).gost($p)) 64/64]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_98 [gost(gost($p).gost($p)) 64/64]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_100 [whirlpool($p) 64/64 OpenSSL]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_101 [whirlpool($s.$p) 64/64 OpenSSL]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_102 [whirlpool($p.$s) 64/64 OpenSSL]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_103 [whirlpool(whirlpool($p)) 64/64 OpenSSL]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_104 [whirlpool(whirlpool_raw($p)) 64/64 OpenSSL]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_105 [whirlpool(whirlpool($p).$s) 64/64 OpenSSL]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_106 [whirlpool($s.whirlpool($p)) 64/64 OpenSSL]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_107 [whirlpool(whirlpool($s).whirlpool($p)) 64/64 OpenSSL]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_108 [whirlpool(whirlpool($p).whirlpool($p)) 64/64 OpenSSL]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_110 [tiger($p) 32/64 sph_tiger]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_111 [tiger($s.$p) 32/64 sph_tiger]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_112 [tiger($p.$s) 32/64 sph_tiger]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_113 [tiger(tiger($p)) 32/64 sph_tiger]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_114 [tiger(tiger_raw($p)) 32/64 sph_tiger]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_115 [tiger(tiger($p).$s) 32/64 sph_tiger]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_116 [tiger($s.tiger($p)) 32/64 sph_tiger]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_117 [tiger(tiger($s).tiger($p)) 32/64 sph_tiger]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_118 [tiger(tiger($p).tiger($p)) 32/64 sph_tiger]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_120 [ripemd128($p) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_121 [ripemd128($s.$p) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_122 [ripemd128($p.$s) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_123 [ripemd128(ripemd128($p)) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_124 [ripemd128(ripemd128_raw($p)) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_125 [ripemd128(ripemd128($p).$s) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_126 [ripemd128($s.ripemd128($p)) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_127 [ripemd128(ripemd128($s).ripemd128($p)) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_128 [ripemd128(ripemd128($p).ripemd128($p)) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_130 [ripemd160($p) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_131 [ripemd160($s.$p) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_132 [ripemd160($p.$s) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_133 [ripemd160(ripemd160($p)) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_134 [ripemd160(ripemd160_raw($p)) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_135 [ripemd160(ripemd160($p).$s) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_136 [ripemd160($s.ripemd160($p)) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_137 [ripemd160(ripemd160($s).ripemd160($p)) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_138 [ripemd160(ripemd160($p).ripemd160($p)) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_140 [ripemd256($p) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_141 [ripemd256($s.$p) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_142 [ripemd256($p.$s) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_143 [ripemd256(ripemd256($p)) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_144 [ripemd256(ripemd256_raw($p)) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_145 [ripemd256(ripemd256($p).$s) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_146 [ripemd256($s.ripemd256($p)) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_147 [ripemd256(ripemd256($s).ripemd256($p)) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_148 [ripemd256(ripemd256($p).ripemd256($p)) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_150 [ripemd320($p) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_151 [ripemd320($s.$p) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_152 [ripemd320($p.$s) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_153 [ripemd320(ripemd320($p)) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_154 [ripemd320(ripemd320_raw($p)) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_155 [ripemd320(ripemd320($p).$s) 32/64 sph_ripemd]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_156 [ripemd320($s.ripemd320($p)) 32/64 sph_ripemd]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_157 [ripemd320(ripemd320($s).ripemd320($p)) 32/64 sph_ripemd]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_158 [ripemd320(ripemd320($p).ripemd320($p)) 32/64 sph_ripemd]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_160 [haval128_3($p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_161 [haval128_3($s.$p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_162 [haval128_3($p.$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_163 [haval128_3(haval128_3($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_164 [haval128_3(haval128_3_raw($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_165 [haval128_3(haval128_3($p).$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_166 [haval128_3($s.haval128_3($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_167 [haval128_3(haval128_3($s).haval128_3($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_168 [haval128_3(haval128_3($p).haval128_3($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_170 [haval128_4($p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_171 [haval128_4($s.$p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_172 [haval128_4($p.$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_173 [haval128_4(haval128_4($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_174 [haval128_4(haval128_4_raw($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_175 [haval128_4(haval128_4($p).$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_176 [haval128_4($s.haval128_4($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_177 [haval128_4(haval128_4($s).haval128_4($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_178 [haval128_4(haval128_4($p).haval128_4($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_180 [haval128_5($p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_181 [haval128_5($s.$p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_182 [haval128_5($p.$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_183 [haval128_5(haval128_5($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_184 [haval128_5(haval128_5_raw($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_185 [haval128_5(haval128_5($p).$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_186 [haval128_5($s.haval128_5($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_187 [haval128_5(haval128_5($s).haval128_5($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_188 [haval128_5(haval128_5($p).haval128_5($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_190 [haval160_3($p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_191 [haval160_3($s.$p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_192 [haval160_3($p.$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_193 [haval160_3(haval160_3($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_194 [haval160_3(haval160_3_raw($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_195 [haval160_3(haval160_3($p).$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_196 [haval160_3($s.haval160_3($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_197 [haval160_3(haval160_3($s).haval160_3($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_198 [haval160_3(haval160_3($p).haval160_3($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_200 [haval160_4($p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_201 [haval160_4($s.$p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_202 [haval160_4($p.$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_203 [haval160_4(haval160_4($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_204 [haval160_4(haval160_4_raw($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_205 [haval160_4(haval160_4($p).$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_206 [haval160_4($s.haval160_4($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_207 [haval160_4(haval160_4($s).haval160_4($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_208 [haval160_4(haval160_4($p).haval160_4($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_210 [haval160_5($p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_211 [haval160_5($s.$p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_212 [haval160_5($p.$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_213 [haval160_5(haval160_5($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_214 [haval160_5(haval160_5_raw($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_215 [haval160_5(haval160_5($p).$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_216 [haval160_5($s.haval160_5($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_217 [haval160_5(haval160_5($s).haval160_5($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_218 [haval160_5(haval160_5($p).haval160_5($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_220 [haval192_3($p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_221 [haval192_3($s.$p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_222 [haval192_3($p.$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_223 [haval192_3(haval192_3($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_224 [haval192_3(haval192_3_raw($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_225 [haval192_3(haval192_3($p).$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_226 [haval192_3($s.haval192_3($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_227 [haval192_3(haval192_3($s).haval192_3($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_228 [haval192_3(haval192_3($p).haval192_3($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_230 [haval192_4($p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_231 [haval192_4($s.$p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_232 [haval192_4($p.$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_233 [haval192_4(haval192_4($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_234 [haval192_4(haval192_4_raw($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_235 [haval192_4(haval192_4($p).$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_236 [haval192_4($s.haval192_4($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_237 [haval192_4(haval192_4($s).haval192_4($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_238 [haval192_4(haval192_4($p).haval192_4($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_240 [haval192_5($p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_241 [haval192_5($s.$p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_242 [haval192_5($p.$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_243 [haval192_5(haval192_5($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_244 [haval192_5(haval192_5_raw($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_245 [haval192_5(haval192_5($p).$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_246 [haval192_5($s.haval192_5($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_247 [haval192_5(haval192_5($s).haval192_5($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_248 [haval192_5(haval192_5($p).haval192_5($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_250 [haval224_3($p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_251 [haval224_3($s.$p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_252 [haval224_3($p.$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_253 [haval224_3(haval224_3($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_254 [haval224_3(haval224_3_raw($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_255 [haval224_3(haval224_3($p).$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_256 [haval224_3($s.haval224_3($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_257 [haval224_3(haval224_3($s).haval224_3($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_258 [haval224_3(haval224_3($p).haval224_3($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_260 [haval224_4($p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_261 [haval224_4($s.$p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_262 [haval224_4($p.$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_263 [haval224_4(haval224_4($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_264 [haval224_4(haval224_4_raw($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_265 [haval224_4(haval224_4($p).$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_266 [haval224_4($s.haval224_4($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_267 [haval224_4(haval224_4($s).haval224_4($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_268 [haval224_4(haval224_4($p).haval224_4($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_270 [haval224_5($p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_271 [haval224_5($s.$p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_272 [haval224_5($p.$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_273 [haval224_5(haval224_5($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_274 [haval224_5(haval224_5_raw($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_275 [haval224_5(haval224_5($p).$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_276 [haval224_5($s.haval224_5($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_277 [haval224_5(haval224_5($s).haval224_5($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_278 [haval224_5(haval224_5($p).haval224_5($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_280 [haval256_3($p) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_281 [haval256_3($s.$p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_282 [haval256_3($p.$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_283 [haval256_3(haval256_3($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_284 [haval256_3(haval256_3_raw($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_285 [haval256_3(haval256_3($p).$s) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_286 [haval256_3($s.haval256_3($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_287 [haval256_3(haval256_3($s).haval256_3($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_288 [haval256_3(haval256_3($p).haval256_3($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_290 [haval256_4($p) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_291 [haval256_4($s.$p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_292 [haval256_4($p.$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_293 [haval256_4(haval256_4($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_294 [haval256_4(haval256_4_raw($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_295 [haval256_4(haval256_4($p).$s) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_296 [haval256_4($s.haval256_4($p)) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_297 [haval256_4(haval256_4($s).haval256_4($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_298 [haval256_4(haval256_4($p).haval256_4($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_300 [haval256_5($p) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_301 [haval256_5($s.$p) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_302 [haval256_5($p.$s) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_303 [haval256_5(haval256_5($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_304 [haval256_5(haval256_5_raw($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_305 [haval256_5(haval256_5($p).$s) 32/64 sph_haval]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_306 [haval256_5($s.haval256_5($p)) 32/64 sph_haval]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_307 [haval256_5(haval256_5($s).haval256_5($p)) 32/64 sph_haval]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_308 [haval256_5(haval256_5($p).haval256_5($p)) 32/64 sph_haval]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_310 [md2($p) 32/64 sph_md2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_311 [md2($s.$p) 32/64 sph_md2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_312 [md2($p.$s) 32/64 sph_md2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_313 [md2(md2($p)) 32/64 sph_md2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_314 [md2(md2_raw($p)) 32/64 sph_md2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_315 [md2(md2($p).$s) 32/64 sph_md2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_316 [md2($s.md2($p)) 32/64 sph_md2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_317 [md2(md2($s).md2($p)) 32/64 sph_md2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_318 [md2(md2($p).md2($p)) 32/64 sph_md2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_320 [panama($p) 32/64 sph_panama]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_321 [panama($s.$p) 32/64 sph_panama]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_322 [panama($p.$s) 32/64 sph_panama]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_323 [panama(panama($p)) 32/64 sph_panama]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_324 [panama(panama_raw($p)) 32/64 sph_panama]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_325 [panama(panama($p).$s) 32/64 sph_panama]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_326 [panama($s.panama($p)) 32/64 sph_panama]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_327 [panama(panama($s).panama($p)) 32/64 sph_panama]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_328 [panama(panama($p).panama($p)) 32/64 sph_panama]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_330 [skein224($p) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_331 [skein224($s.$p) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_332 [skein224($p.$s) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_333 [skein224(skein224($p)) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_334 [skein224(skein224_raw($p)) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_335 [skein224(skein224($p).$s) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_336 [skein224($s.skein224($p)) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_337 [skein224(skein224($s).skein224($p)) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_338 [skein224(skein224($p).skein224($p)) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_340 [skein256($p) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_341 [skein256($s.$p) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_342 [skein256($p.$s) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_343 [skein256(skein256($p)) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_344 [skein256(skein256_raw($p)) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_345 [skein256(skein256($p).$s) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_346 [skein256($s.skein256($p)) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_347 [skein256(skein256($s).skein256($p)) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_348 [skein256(skein256($p).skein256($p)) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_350 [skein384($p) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_351 [skein384($s.$p) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_352 [skein384($p.$s) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_353 [skein384(skein384($p)) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_354 [skein384(skein384_raw($p)) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_355 [skein384(skein384($p).$s) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_356 [skein384($s.skein384($p)) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_357 [skein384(skein384($s).skein384($p)) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_358 [skein384(skein384($p).skein384($p)) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_360 [skein512($p) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_361 [skein512($s.$p) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_362 [skein512($p.$s) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_363 [skein512(skein512($p)) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_364 [skein512(skein512_raw($p)) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_365 [skein512(skein512($p).$s) 32/64 sph_skein]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_366 [skein512($s.skein512($p)) 32/64 sph_skein]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_367 [skein512(skein512($s).skein512($p)) 32/64 sph_skein]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_368 [skein512(skein512($p).skein512($p)) 32/64 sph_skein]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_370 [sha3_224($p) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_371 [sha3_224($s.$p) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_372 [sha3_224($p.$s) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_373 [sha3_224(sha3_224($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_374 [sha3_224(sha3_224_raw($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_375 [sha3_224(sha3_224($p).$s) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_376 [sha3_224($s.sha3_224($p)) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_377 [sha3_224(sha3_224($s).sha3_224($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_378 [sha3_224(sha3_224($p).sha3_224($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_380 [sha3_256($p) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_381 [sha3_256($s.$p) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_382 [sha3_256($p.$s) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_383 [sha3_256(sha3_256($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_384 [sha3_256(sha3_256_raw($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_385 [sha3_256(sha3_256($p).$s) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_386 [sha3_256($s.sha3_256($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_387 [sha3_256(sha3_256($s).sha3_256($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_388 [sha3_256(sha3_256($p).sha3_256($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_390 [sha3_384($p) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_391 [sha3_384($s.$p) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_392 [sha3_384($p.$s) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_393 [sha3_384(sha3_384($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_394 [sha3_384(sha3_384_raw($p)) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_395 [sha3_384(sha3_384($p).$s) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_396 [sha3_384($s.sha3_384($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_397 [sha3_384(sha3_384($s).sha3_384($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_398 [sha3_384(sha3_384($p).sha3_384($p)) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_400 [sha3_512($p) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_401 [sha3_512($s.$p) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_402 [sha3_512($p.$s) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_403 [sha3_512(sha3_512($p)) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_404 [sha3_512(sha3_512_raw($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_405 [sha3_512(sha3_512($p).$s) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_406 [sha3_512($s.sha3_512($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_407 [sha3_512(sha3_512($s).sha3_512($p)) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_408 [sha3_512(sha3_512($p).sha3_512($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_410 [keccak_256($p) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_411 [keccak_256($s.$p) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_412 [keccak_256($p.$s) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_413 [keccak_256(keccak_256($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_414 [keccak_256(keccak_256_raw($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_415 [keccak_256(keccak_256($p).$s) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_416 [keccak_256($s.keccak_256($p)) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_417 [keccak_256(keccak_256($s).keccak_256($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_418 [keccak_256(keccak_256($p).keccak_256($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_420 [keccak_512($p) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_421 [keccak_512($s.$p) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_422 [keccak_512($p.$s) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_423 [keccak_512(keccak_512($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_424 [keccak_512(keccak_512_raw($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_425 [keccak_512(keccak_512($p).$s) 64/64 keccak]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_426 [keccak_512($s.keccak_512($p)) 64/64 keccak]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_427 [keccak_512(keccak_512($s).keccak_512($p)) 64/64 keccak]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_428 [keccak_512(keccak_512($p).keccak_512($p)) 64/64 keccak]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1001 [md5(md5(md5(md5($p)))) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1002 [md5(md5(md5(md5(md5($p))))) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1003 [md5(md5($p).md5($p)) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1004 [md5(md5(md5(md5(md5(md5($p)))))) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1005 [md5(md5(md5(md5(md5(md5(md5($p))))))) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1006 [md5(md5(md5(md5(md5(md5(md5(md5($p)))))))) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1007 [md5(md5($p).$s) (vBulletin) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1008 [md5($p.$s) (RADIUS User-Password) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1009 [md5($s.$p) (RADIUS Responses) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1010 [md5($p null_padded_to_len_100) RAdmin v2.x MD5 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1011 [md5($p.md5($s)) (WebEdition CMS) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1012 [md5($p.md5($s)) (WebEdition CMS) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1013 [md5($p.PMD5(username)) (WebEdition CMS) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1014 [md5($p.$s) (long salt) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1015 [md5(md5($p.$u).$s) (PostgreSQL 'pass the hash') 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1016 [md5($p.$s) (long salt) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1017 [md5($s.$p) (long salt) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1018 [md5(sha1(sha1($p))) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1019 [md5(sha1(sha1(md5($p)))) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1020 [md5(sha1(md5($p))) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1021 [md5(sha1(md5(sha1($p)))) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1022 [md5(sha1(md5(sha1(md5($p))))) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1023 [sha1($p) (hash truncated to length 32) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1024 [sha1(md5($p)) (hash truncated to length 32) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1025 [sha1(md5(md5($p))) (hash truncated to length 32) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1026 [sha1(sha1($p)) (hash truncated to length 32) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1027 [sha1(sha1(sha1($p))) (hash truncated to length 32) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1028 [sha1(sha1_raw($p)) (hash truncated to length 32) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1029 [sha256($p) (hash truncated to length 32) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1030 [whirlpool($p) (hash truncated to length 32) 64/64 OpenSSL]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1031 [gost($p) (hash truncated to length 32) 64/64]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1032 [sha1_64(utf16($p)) (PeopleSoft) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1033 [sha1_64(utf16($p).$s) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1034 [md5($p.$u) (PostgreSQL MD5) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1300 [md5(md5_raw($p)) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1350 [md5(md5($s.$p):$s) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1400 [sha1(utf16($p)) (Microsoft CREDHIST) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1401 [md5($u.\nskyper\n.$p) (Skype MD5) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1501 [sha1($s.sha1($p)) (Redmine) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1502 [sha1(sha1($p).$s) (XenForo SHA-1) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1503 [sha256(sha256($p).$s) (XenForo SHA-256) 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1504 [sha1($s.$p.$s) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1505 [md5($p.$s.md5($p.$s)) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1506 [md5($u.:XDB:.$p) (Oracle 12c "H" hash) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1507 [sha1(utf16($const.$p)) (Mcafee master pass) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1518 [md5(sha1($p).md5($p).sha1($p)) 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1550 [md5($u.:mongo:.$p) (MONGODB-CR system hash) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1551 [md5($s.$u.(md5($u.:mongo:.$p)) (MONGODB-CR network hash) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1552 [md5($s.$u.(md5($u.:mongo:.$p)) (MONGODB-CR network hash) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1560 [md5($s.$p.$s2) [SocialEngine] 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_1588 [sha256($s.sha1($p)) (ColdFusion 11) 128/128 SSE4.1 4x]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_1590 [sha1(utf16be(space_pad_10(uc($s)).$p)) (IBM AS/400 SHA1) 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_1592 [sha1($s.sha1($s.sha1($p))) 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_1600 [sha1($s.utf16le($p)) [Oracle PeopleSoft PS_TOKEN] 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_2000 [md5($p) (PW > 55 bytes) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_2001 [md5($p.$s) (joomla) (PW > 23 bytes) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_2002 [md5(md5($p)) (e107) (PW > 55 bytes) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_2003 [md5(md5(md5($p))) (PW > 55 bytes) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_2004 [md5($s.$p) (OSC) (PW > 31 bytes) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_2005 [md5($s.$p.$s) (PW > 31 bytes) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_2006 [md5(md5($p).$s) (PW > 55 bytes) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_2008 [md5(md5($s).$p) (PW > 23 bytes) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: dynamic_2009 [md5($s.md5($p)) (salt > 23 bytes) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dynamic_2010 [md5($s.md5($s.$p)) (PW > 32 or salt > 23 bytes) 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: dynamic_2011 [md5($s.md5($p.$s)) (PW > 32 or salt > 23 bytes) 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-\|/-\|/-\|/-PASS

Testing: dynamic_2014 [md5($s.md5($p).$s) (PW > 55 or salt > 11 bytes) 128/128 SSE4.1 4x5]... \|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: agilekeychain, 1Password Agile Keychain [PBKDF2-SHA1 AES 128/128 SSE4.1 4x2]... |/-\|PASS

Testing: aix-ssha1, AIX LPA {ssha1} [PBKDF2-SHA1 128/128 SSE4.1 4x2]... /-\|PASS

Testing: aix-ssha256, AIX LPA {ssha256} [PBKDF2-SHA256 128/128 SSE4.1 4x]... /-\|PASS

Testing: aix-ssha512, AIX LPA {ssha512} [PBKDF2-SHA512 128/128 SSE4.1 2x]... /-\|/-\PASS

Testing: argon2 [Blake2 SSSE3]... |/-\|/-\|PASS

Testing: as400-des, AS/400 DES [DES 32/64]... /-\|/-\|/-\|/-\|PASS

Testing: as400-ssha1, AS400-SaltedSHA1 [sha1(utf16be(space_pad_10(uc($s)).$p)) (IBM AS/400 SHA1) 128/128 SSE4.1 4x2]... /-\|/-\|/-PASS

Testing: asa-md5, Cisco ASA [md5($p.$s) (Cisco ASA) 128/128 SSE4.1 4x5]... \|/-\|/PASS

Testing: axcrypt, AxCrypt [SHA1 AES 32/64]... -\|/-\PASS

Testing: AzureAD [PBKDF2-SHA256 128/128 SSE4.1 4x]... |/-\|PASS

Testing: BestCrypt [Jetico BestCrypt (.jbc) PKCS12 PBE (Whirlpool / SHA-1 to SHA-512) 32/64]... /-\|/-\|/-PASS

Testing: bfegg, Eggdrop [Blowfish 32/64]... \|/-\|/-\PASS

Testing: Bitcoin, Bitcoin Core [SHA512 AES 128/128 SSE4.1 2x]... |/-\|/-PASS

Testing: BitLocker, BitLocker [SHA-256 AES 64/64]... \|/-\|PASS

Testing: BKS [PKCS12 PBE 128/128 SSE4.1 4x2]... /-\|/PASS

Testing: Blackberry-ES10 (101x) [SHA-512 128/128 SSE4.1 2x]... -\|/-PASS

Testing: WoWSRP, Battlenet [SHA1 32/64 GMP-exp]... \|/-\|PASS

Testing: Blockchain, My Wallet (x10) [PBKDF2-SHA1 AES 128/128 SSE4.1 4x2]... /-\|/-PASS

Testing: chap, iSCSI CHAP authentication / EAP-MD5 [MD5 32/64]... \|/-\|PASS

Testing: Clipperz, SRP [SHA256 32/64 GMP-exp]... /-\|PASS

Testing: cloudkeychain, 1Password Cloud Keychain [PBKDF2-SHA512 128/128 SSE4.1 2x]... /-\PASS

Testing: dynamic=md5($p) [128/128 SSE4.1 4x5]... |/-\|/PASS

Testing: cq, ClearQuest [CQWeb]... -\|/-PASS

Testing: CRC32 [CRC32 32/64 CRC-32C 32/64]... \|/-\|/-\PASS

Testing: sha1crypt, NetBSD's sha1crypt [PBKDF1-SHA1 128/128 SSE4.1 4x2]... |/-\|PASS

Testing: sha256crypt, crypt(3) $5$ (rounds=5000) [SHA256 128/128 SSE4.1 4x]... /-\|/-\|PASS

Testing: sha512crypt, crypt(3) $6$ (rounds=5000) [SHA512 128/128 SSE4.1 2x]... /-\|/-\PASS

Testing: Citrix_NS10, Netscaler 10 [SHA1 128/128 SSE4.1 4x2]... |/-\|/-\|PASS

Testing: dahua, "MD5 based authentication" Dahua [MD5 32/64]... /-\|/-\PASS

Testing: Django (x10000) [PBKDF2-SHA256 128/128 SSE4.1 4x]... |/-\|/-PASS

Testing: django-scrypt [Salsa20/8 128/128 SSE2]... \|/PASS

Testing: dmd5, DIGEST-MD5 C/R [MD5 32/64]... -\PASS

Testing: dmg, Apple DMG [PBKDF2-SHA1 128/128 SSE4.1 4x2 3DES/AES]... |/-\|/-\PASS

Testing: dominosec, Lotus Notes/Domino 6 More Secure Internet Password [8/64]... |/-\|/-\|PASS

Testing: dominosec8, Lotus Notes/Domino 8 [8/64]... /-\PASS

Testing: DPAPImk, DPAPI masterkey file v1 and v2 [SHA1/MD4 PBKDF2-(SHA1/SHA512)-DPAPI-variant 3DES/AES256 128/128 SSE4.1 4x2]... |/-\|/-PASS

Testing: dragonfly3-32, DragonFly BSD $3$ w/ bug, 32-bit [SHA256 32/64 OpenSSL]... \|/-\|/-PASS

Testing: dragonfly3-64, DragonFly BSD $3$ w/ bug, 64-bit [SHA256 32/64 OpenSSL]... \|/-\|/-PASS

Testing: dragonfly4-32, DragonFly BSD $4$ w/ bugs, 32-bit [SHA512 64/64 OpenSSL]... \|/-\|PASS

Testing: dragonfly4-64, DragonFly BSD $4$ w/ bugs, 64-bit [SHA512 64/64 OpenSSL]... /-\|/-PASS

Testing: Drupal7, $S$ (x16385) [SHA512 128/128 SSE4.1 2x]... \|/-PASS

Testing: eCryptfs (65536x) [SHA512 128/128 SSE4.1 2x]... \|/-\|PASS

Testing: eigrp, EIGRP MD5 / HMAC-SHA-256 authentication [MD5 32/64]... /-\|/-\PASS

Testing: electrum, Electrum Wallet [SHA256 AES / PBKDF2-SHA512 128/128 SSE4.1 4x2]... |/-\|/-\|/PASS

Testing: EncFS [PBKDF2-SHA1 128/128 SSE4.1 4x2 AES/Blowfish]... -\|/-PASS

Testing: enpass, Enpass Password Manager [PBKDF2-SHA1 128/128 SSE4.1 4x2]... \|PASS

Testing: EPI, EPiServer SID [SHA1 32/64]... /-\|/PASS

Testing: EPiServer [SHA1/SHA256 128/128 SSE4.1 4x]... -\|/-\|/-\PASS

Testing: ethereum, Ethereum Wallet [PBKDF2-SHA256/scrypt Keccak 128/128 SSE4.1 4x]... |/-\|/-PASS

Testing: fde, Android FDE [PBKDF2-SHA1 128/128 SSE4.1 4x2 SHA256/AES]... \|PASS

Testing: Fortigate, FortiOS [SHA1 128/128 SSE4.1 4x2]... /-\|PASS

Testing: FormSpring [sha256($s.$p) 128/128 SSE4.1 4x]... /-\|/PASS

Testing: FVDE, FileVault 2 [PBKDF2-SHA256 AES 128/128 SSE4.1 4x]... -\|PASS

Testing: geli, FreeBSD GELI [PBKDF2-SHA512 128/128 SSE4.1 4x2]... /-\PASS

Testing: gost, GOST R 34.11-94 [64/64]... |/-\|/-\|/-PASS

Testing: gpg, OpenPGP / GnuPG Secret Key [32/64]... \|/-\|/-\|/-\|/-\PASS

Testing: HAVAL-128-4 [32/64]... |/-\|/-\|PASS

Testing: HAVAL-256-3 [32/64]... /-\|/-\|/-PASS

Testing: hdaa, HTTP Digest access authentication [MD5 128/128 SSE4.1 4x5]... \|/-\|PASS

Testing: HMAC-MD5 [password is key, MD5 128/128 SSE4.1 4x5]... /-\|/-\|/-\|/PASS

Testing: HMAC-SHA1 [password is key, SHA1 128/128 SSE4.1 4x2]... -\|/-\|/PASS

Testing: HMAC-SHA224 [password is key, SHA224 128/128 SSE4.1 4x]... -\|PASS

Testing: HMAC-SHA256 [password is key, SHA256 128/128 SSE4.1 4x]... /-\|/-PASS

Testing: HMAC-SHA384 [password is key, SHA384 128/128 SSE4.1 2x]... \|/-PASS

Testing: HMAC-SHA512 [password is key, SHA512 128/128 SSE4.1 2x]... \|/-\|/-\|/-\|/-PASS

Testing: hMailServer [sha256($s.$p) 128/128 SSE4.1 4x]... \|/-\PASS

Testing: hsrp, "MD5 authentication" HSRP, HSRPv2, VRRP, GLBP [MD5 32/64]... |/-\|/PASS

Testing: IKE, PSK [HMAC MD5/SHA1 32/64]... -\|PASS

Testing: ipb2, Invision Power Board 2.x [MD5 128/128 SSE4.1 4x5]... /-\|/PASS

Testing: itunes-backup, Apple iTunes Backup [PBKDF2-SHA1 AES 128/128 SSE4.1 4x2]... -\|/-PASS

Testing: iwork, Apple iWork '09 / '13 / '14 [PBKDF2-SHA1 AES 128/128 SSE4.1 4x2]... \|/-\PASS

Testing: KeePass [SHA256 AES 32/64 OpenSSL]... |/-\|/-\|/PASS

Testing: keychain, Mac OS X Keychain [PBKDF2-SHA1 3DES 128/128 SSE4.1 4x2]... -\|/-\|PASS

Testing: keyring, GNOME Keyring [SHA256 AES 128/128 SSE4.1 4x]... /-\PASS

Testing: keystore, Java KeyStore [SHA1 128/128 SSE4.1 4x2]... |/-\PASS

Testing: known_hosts, HashKnownHosts HMAC-SHA1 [SHA1 32/64]... |/-\|PASS

Testing: krb4, Kerberos v4 TGT [DES 32/64]... /-\|/-\|/-\PASS

Testing: krb5, Kerberos v5 TGT [3DES 32/64]... |/-PASS

Testing: krb5pa-sha1, Kerberos 5 AS-REQ Pre-Auth etype 17/18 [PBKDF2-SHA1 128/128 SSE4.1 4x2]... \|/-\|/-PASS

Testing: krb5tgs, Kerberos 5 TGS etype 23 [MD4 HMAC-MD5 RC4]... \|/-\PASS

Testing: krb5-18, Kerberos 5 db etype 18 [PBKDF2-SHA1 128/128 SSE4.1 4x2 AES]... |/-PASS

Testing: kwallet, KDE KWallet [SHA1 / PBKDF2-SHA512 128/128 SSE4.1 4x2]... \|/-PASS

Testing: lp, LastPass offline [PBKDF2-SHA256 128/128 SSE4.1 4x]... \|/-\PASS

Testing: leet [SHA-512(128/128 SSE4.1 2x) + Whirlpool(OpenSSL/64)]... |/-\|PASS

Testing: lotus5, Lotus Notes/Domino 5 [8/64 X3]... /-\|/PASS

Testing: lotus85, Lotus Notes/Domino 8.5 [8/64]... -\|PASS

Testing: LUKS [PBKDF2-SHA1 128/128 SSE4.1 4x2]... /-\PASS

Testing: MD2 [MD2 32/64]... |/-\PASS

Testing: mdc2, MDC-2 [MDC-2DES]... |/-\PASS

Testing: MediaWiki [md5($s.md5($p)) 128/128 SSE4.1 4x5]... |/-\|/-PASS

Testing: money, Microsoft Money (2002 to Money Plus) [MD5/SHA1 32/64]... \|/-\|/-\|/PASS

Testing: MongoDB, system / network [MD5 32/64]... -\|/-\|/-\|/-PASS

Testing: scram [SCRAM PBKDF2-SHA1 128/128 SSE4.1 4x2]... \|/PASS

Testing: Mozilla, Mozilla key3.db [SHA1 3DES 32/64]... -\|/PASS

Testing: mscash, MS Cache Hash (DCC) [MD4 32/64]... -\|/-\|/-\|PASS

Testing: mscash2, MS Cache Hash 2 (DCC2) [PBKDF2-SHA1 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-PASS

Testing: MSCHAPv2, C/R [MD4 DES (ESS MD5) 128/128 SSE4.1 4x4]... \|/-\|/-\|/-\|/-PASS

Testing: mschapv2-naive, MSCHAPv2 C/R [MD4 DES DES 128/128 SSE2-16 naive]... \|/-\|/-\|/-\|/-\|PASS

Testing: krb5pa-md5, Kerberos 5 AS-REQ Pre-Auth etype 23 [32/64]... /-\|/-\|/-\|PASS

Testing: mssql, MS SQL [SHA1 128/128 SSE4.1 4x2]... /-\|/PASS

Testing: mssql05, MS SQL 2005 [SHA1 128/128 SSE4.1 4x2]... -\|/-\|PASS

Testing: mssql12, MS SQL 2012/2014 [SHA512 128/128 SSE4.1 2x]... /-\|/-\|/-\|/PASS

Testing: multibit, MultiBit Wallet [MD5 AES 32/64]... -\|/PASS

Testing: mysqlna, MySQL Network Authentication [SHA1 32/64]... -\|/PASS

Testing: mysql-sha1, MySQL 4.1+ [SHA1 128/128 SSE4.1 4x2]... -\|/-\|/-\|/-\PASS

Testing: mysql, MySQL pre-4.1 [32/64]... |/-\|/-\|/-\|/-\PASS

Testing: net-ah, IPsec AH HMAC-MD5-96 [MD5 32/64]... |/-\|PASS

Testing: nethalflm, HalfLM C/R [DES 32/64]... /-\|/-\|PASS

Testing: netlm, LM C/R [DES 32/64]... /-\|/-\|/-PASS

Testing: netlmv2, LMv2 C/R [MD4 HMAC-MD5 32/64]... \|/-\|/-\PASS

Testing: net-md5, "Keyed MD5" RIPv2, OSPF, BGP, SNMPv2 [MD5 32/64]... |/-\|/-\|/-\|PASS

Testing: netntlmv2, NTLMv2 C/R [MD4 HMAC-MD5 32/64]... /-\|/-\|/-\|/-\PASS

Testing: netntlm, NTLMv1 C/R [MD4 DES (ESS MD5) 128/128 SSE4.1 4x4]... |/-\|/-\|/-\PASS

Testing: netntlm-naive, NTLMv1 C/R [MD4 DES (ESS MD5) DES 128/128 SSE2-16 naive]... |/-\|/-\|/-\|/PASS

Testing: net-sha1, "Keyed SHA1" BFD [SHA1 32/64]... -\|PASS

Testing: nk, Nuked-Klan CMS [SHA1 MD5 32/64]... /-\|/-PASS

Testing: md5ns, Netscreen [md5($s.$p) (OSC) (PW > 31 bytes) 128/128 SSE4.1 4x5]... \|/-PASS

Testing: nsec3, DNSSEC NSEC3 [32/64]... \|/PASS

Testing: NT [MD4 128/128 SSE4.1 4x4]... -\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: o10glogon, Oracle 10g-logon protocol [DES-AES128-MD5 32/64]... -\|/PASS

Testing: o3logon, Oracle O3LOGON protocol [SHA1 DES 32/64]... -\|/-PASS

Testing: o5logon, Oracle O5LOGON protocol [SHA1 AES 32/64]... \|/-\|/-\|/PASS

Testing: ODF [SHA1/SHA256 128/128 SSE4.1 4x2 BF/AES]... -\|/-\|PASS

Testing: Office, 2007/2010/2013 [SHA1 128/128 SSE4.1 4x2 / SHA512 128/128 SSE4.1 2x AES]... /-\|/-\|/-\|/-\|/-\|PASS

Testing: oldoffice, MS Office <= 2003 [MD5/SHA1 RC4 32/64]... /-\|/-\|/-\|PASS

Testing: OpenBSD-SoftRAID (8192 iterations) [PBKDF2-SHA1 128/128 SSE4.1 4x2]... /-\|PASS

Testing: openssl-enc, OpenSSL "enc" encryption [32/64]... /-\|/-\PASS

Testing: oracle, Oracle 10 [DES 32/64]... |/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: oracle11, Oracle 11g [SHA1 128/128 SSE4.1 4x2]... /-\|/-PASS

Testing: Oracle12C [PBKDF2-SHA512 128/128 SSE4.1 2x]... \|/PASS

Testing: osc, osCommerce [md5($s.$p) (OSC) 128/128 SSE4.1 4x5]... -\|/-PASS

Testing: ospf, OSPF / IS-IS [HMAC-SHA-X 32/64]... \|/-\|/PASS

Testing: Padlock [PBKDF2-SHA256 AES 128/128 SSE4.1 4x]... -\|/PASS

Testing: Palshop, MD5(Palshop) [MD5 + SHA1 32/64]... -\|/PASS

Testing: Panama [Panama 32/64]... -\|/-\|/-PASS

Testing: PBKDF2-HMAC-MD4 [PBKDF2-MD4 128/128 SSE4.1 4x4]... \|/-\|/-\|/-\|/PASS

Testing: PBKDF2-HMAC-MD5 [PBKDF2-MD5 128/128 SSE4.1 4x5]... -\|/-\|/-\|/-PASS

Testing: PBKDF2-HMAC-SHA1 [PBKDF2-SHA1 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|/PASS

Testing: PBKDF2-HMAC-SHA256 [PBKDF2-SHA256 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: PBKDF2-HMAC-SHA512, GRUB2 / OS X 10.8+ [PBKDF2-SHA512 128/128 SSE4.1 2x]... /-\|/-\|/-\|PASS

Testing: PDF [MD5 SHA2 RC4/AES 32/64]... /-\|/-\|/-\|/-\|/PASS

Testing: PEM, PKCS#8 private key (RSA/DSA/ECDSA) [PBKDF2-SHA1 3DES 128/128 SSE4.1 4x2]... -\|/-\PASS

Testing: pfx [PKCS12 PBE (.pfx, .p12) (SHA-1 to SHA-512) 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\PASS

Testing: phpass ($P$9) [phpass ($P$ or $H$) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\|/-PASS

Testing: PHPS [md5(md5($p).$s) 128/128 SSE4.1 4x5]... \|/-\PASS

Testing: PHPS2 [md5(md5($p).$s) 128/128 SSE4.1 4x5]... |/-\PASS

Testing: pix-md5, Cisco PIX [md5($p) (Cisco PIX) 128/128 SSE4.1 4x5]... |/-\|/-\|/PASS

Testing: PKZIP [32/64]... -\|/-\|/-\|PASS

Testing: po, Post.Office [MD5 32/64]... /-\|/PASS

Testing: pomelo [POMELO 128/128 SSE2 1x]... -\|PASS

Testing: postgres, PostgreSQL C/R [MD5 32/64]... /-\|/-\|/PASS

Testing: PST, custom CRC-32 [32/64]... -\|/-\|/PASS

Testing: PuTTY, Private Key [SHA1/AES 32/64]... -\|PASS

Testing: pwsafe, Password Safe [SHA256 128/128 SSE4.1 4x]... /-\|/-PASS

Testing: qnx, qnx hash (rounds=1000) [QNX 32/64 generic]... \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: RACF [DES 32/64]... /-\|/-\|/-\PASS

Testing: RAdmin, v2.x [MD5 32/64]... |/-\|/-\|PASS

Testing: RAKP, IPMI 2.0 RAKP (RMCP+) [HMAC-SHA1 128/128 SSE4.1 4x2]... /-\|/PASS

Testing: rar, RAR3 (4 characters) [SHA1 128/128 SSE4.1 4x2 AES]... -\|/-\PASS

Testing: RAR5 [PBKDF2-SHA256 128/128 SSE4.1 4x]... |/-\|/PASS

Testing: Raw-SHA512 [SHA512 128/128 SSE4.1 2x]... -\|/-\|/-\|/-\PASS

Testing: Raw-Blake2 [BLAKE2b 512 128/128 SSE4.1]... |/-\|/-\PASS

Testing: Raw-Keccak [Keccak 512 32/64]... |/-\|/PASS

Testing: Raw-Keccak-256 [Keccak 256 32/64]... -\|/-\PASS

Testing: Raw-MD4 [MD4 128/128 SSE4.1 4x4]... |/-\|/-\|/-\|/-\PASS

Testing: Raw-MD5 [MD5 128/128 SSE4.1 4x5]... |/-\|/-\PASS

Testing: Raw-MD5u [md5(utf16($p)) 128/128 SSE4.1 4x5]... |/-\|/-\|/-\PASS

Testing: Raw-SHA1 [SHA1 128/128 SSE4.1 4x2]... |/-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: Raw-SHA1-AxCrypt [SHA1 128/128 SSE4.1 4x2]... -\|/-\|/-PASS

Testing: Raw-SHA1-Linkedin [SHA1 128/128 SSE4.1 4x2]... \|/-\|/-\PASS

Testing: Raw-SHA224 [SHA224 128/128 SSE4.1 4x]... |/-\|/PASS

Testing: Raw-SHA256 [SHA256 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: Raw-SHA256-ng [SHA256 128/128 SSE4.1 4x]... -\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: Raw-SHA3 [SHA3 512 32/64]... -\|/PASS

Testing: Raw-SHA384 [SHA384 128/128 SSE4.1 2x]... -\|/-\PASS

Testing: Raw-SHA512-ng [SHA512 128/128 SSSE3 2x]... |/-\|/-\|/-\|/PASS

Testing: ripemd-128, RIPEMD 128 [32/64]... -\|/-\|/-\|PASS

Testing: ripemd-160, RIPEMD 160 [32/64]... /-\|/-\|/-\PASS

Testing: rsvp, HMAC-MD5 / HMAC-SHA1, RSVP, IS-IS [MD5 32/64]... |/-\|/PASS

Testing: Siemens-S7 [HMAC-SHA1 32/64]... -\|PASS

Testing: Salted-SHA1 [SHA1 128/128 SSE4.1 4x2]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: SSHA512, LDAP [SHA512 128/128 SSE4.1 2x]... -\|/PASS

Testing: sapb, SAP CODVN B (BCODE) [MD5 128/128 SSE4.1 4x5]... -\|/-\|/-\|PASS

Testing: sapg, SAP CODVN F/G (PASSCODE) [SHA1 128/128 SSE4.1 4x2]... /-\|/-\|/-PASS

Testing: saph, SAP CODVN H (PWDSALTEDHASH) (SHA1x1024) [SHA-1/SHA-2 128/128 SSE4.1 4x2]... \|/-\|/-\|/-\|PASS

Testing: 7z, 7-Zip (512K iterations) [SHA256 128/128 SSE4.1 4x AES]... /-\|/-\|/-\|/-PASS

Testing: Raw-SHA1-ng, (pwlen <= 15) [SHA1 128/128 SSE4.1 4x]... \|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: SIP [MD5 32/64]... /-\|/-\PASS

Testing: skein-256, Skein 256 [Skein 32/64]... |/-\|/-PASS

Testing: skein-512, Skein 512 [Skein 32/64]... \|/-\|/PASS

Testing: skey, S/Key [MD4/MD5/SHA1/RMD160 32/64]... -\|/-PASS

Testing: SL3, Nokia operator unlock [SHA1 128/128 SSE4.1 4x2]... \|/PASS

Testing: aix-smd5, AIX LPA {smd5} (modified crypt-md5) [MD5 32/64]... -\|/-\|PASS

Testing: Snefru-128 [32/64]... /-\PASS

Testing: Snefru-256 [32/64]... |/-PASS

Testing: LastPass, sniffed sessions [PBKDF2-SHA256 AES 128/128 SSE4.1 4x]... \|/-PASS

Testing: SNMP, SNMPv3 USM [HMAC-MD5-96/HMAC-SHA1-96 32/64]... \|/-\|/PASS

Testing: SSH-ng [RSA/DSA/EC/OPENSSH (SSH private keys) 32/64]... -\|/-\|/-\|PASS

Testing: Stribog-256 [GOST R 34.11-2012 128/128 SSE4.1 1x]... /-\PASS

Testing: Stribog-512 [GOST R 34.11-2012 128/128 SSE4.1 1x]... |/-PASS

Testing: STRIP, Password Manager [PBKDF2-SHA1 128/128 SSE4.1 4x2]... \|/PASS

Testing: SunMD5 [MD5 128/128 SSE4.1 4x5]... -\|/-\|/-PASS

Testing: sxc, StarOffice .sxc [SHA1 128/128 SSE4.1 4x2 Blowfish]... \|/-\PASS

Testing: SybaseASE, Sybase ASE [SHA256 128/128 SSE4.1 4x]... |/-PASS

Testing: Sybase-PROP [salted FEAL-8 32/64]... \|/PASS

Testing: tcp-md5, TCP MD5 Signatures, BGP, MSDP [MD5 32/64]... -\|PASS

Testing: Tiger [Tiger 32/64]... /-\|/PASS

Testing: tc_aes_xts, TrueCrypt AES256_XTS [SHA512 128/128 SSE4.1 2x /RIPEMD160/WHIRLPOOL]... -\|/-\|/PASS

Testing: tc_ripemd160, TrueCrypt AES256_XTS [RIPEMD160 32/64]... -\|PASS

Testing: tc_sha512, TrueCrypt AES256_XTS [SHA512 128/128 SSE4.1 2x]... /-\|PASS

Testing: tc_whirlpool, TrueCrypt AES256_XTS [WHIRLPOOL 64/64]... /-\PASS

Testing: vdi, VirtualBox-VDI AES_XTS [PBKDF2-SHA256 128/128 SSE4.1 4x + AES_XTS]... |/-\|/-\|PASS

Testing: OpenVMS, Purdy [32/64]... /-\|PASS

Testing: VNC [DES 32/64]... /-\|/-\|/PASS

Testing: vtp, "MD5 based authentication" VTP [MD5 32/64]... -\|PASS

Testing: wbb3, WoltLab BB3 [SHA1 32/64]... /-\|/-\|PASS

Testing: whirlpool [WHIRLPOOL 32/64]... /-\PASS

Testing: whirlpool0 [WHIRLPOOL-0 32/64]... |/-PASS

Testing: whirlpool1 [WHIRLPOOL-1 32/64]... \|/PASS

Testing: wpapsk, WPA/WPA2 PSK [PBKDF2-SHA1 128/128 SSE4.1 4x2]... -\|/-PASS

Testing: xmpp-scram [XMPP SCRAM PBKDF2-SHA1 128/128 SSE4.1 4x2]... \|/-PASS

Testing: xsha, Mac OS X 10.4 - 10.6 [SHA1 128/128 SSE4.1 4x2]... \|/-\|/-\|PASS

Testing: xsha512, Mac OS X 10.7 [SHA512 128/128 SSE4.1 2x]... /-\|/-\|/-\|/-\|/-\|PASS

Testing: ZIP, WinZip [PBKDF2-SHA1 128/128 SSE4.1 4x2]... /-\PASS

Testing: ZipMonster, MD5(ZipMonster) [MD5-128/128 SSE4.1 4x5 x 50000]... |/-PASS

Testing: plaintext, $0$ [n/a]... \|/-\|PASS

Testing: has-160 [HAS-160 32/64]... /-\|/-\|PASS

Testing: NT-old [MD4 128/128 X2 SSE2-16]... /-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: dummy [N/A]... /-\|/-PASS

Testing: crypt, generic crypt(3) DES [?/64]... \|/-\|PASS

All 680 formats passed self-tests!

Device 0: Intel(R) Xeon(R) CPU           X5570  @ 2.93GHz

Testing: sha1crypt-opencl, (NetBSD) [PBKDF1-SHA1 OpenCL 4x]... /-\|/PASS

Testing: oldoffice-opencl, MS Office <= 2003 [MD5/SHA1 RC4 OpenCL]... -\|/-\|PASS

Testing: PBKDF2-HMAC-MD4-opencl [PBKDF2-MD4 OpenCL 4x]... /-\|/-\|/-\|/-\PASS

Testing: PBKDF2-HMAC-MD5-opencl [PBKDF2-MD5 OpenCL 4x]... |/-\|/-\|/-\|PASS

Testing: PBKDF2-HMAC-SHA1-opencl [PBKDF2-SHA1 OpenCL 4x]... /-\|/-\|/-\|/-\PASS

Testing: rar-opencl, RAR3 (length 4) [SHA1 OpenCL AES]... |/-\|/PASS

Testing: RAR5-opencl [PBKDF2-SHA256 OpenCL]... -\|/-\PASS

Testing: truecrypt-opencl, TrueCrypt AES256_XTS [RIPEMD160 OpenCL]... |/-PASS

Testing: lotus5-opencl, Lotus Notes/Domino 5 [OpenCL]... \|/-\PASS

Testing: agilekeychain-opencl, 1Password Agile Keychain [PBKDF2-SHA1 AES OpenCL]... |/-\|PASS

Testing: bcrypt-opencl ("$2a$05", 32 iterations) [Blowfish OpenCL]... /-\|/-\|/-\|/-\PASS

Testing: BitLocker-opencl, BitLocker [SHA256 AES OpenCL]... |/-\PASS

Testing: blockchain-opencl, blockchain My Wallet [PBKDF2-SHA1 OpenCL AES]... |/-\|/PASS

Testing: md5crypt-opencl, crypt(3) $1$ [MD5 OpenCL]... -\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: sha256crypt-opencl, crypt(3) $5$ (rounds=5000) [SHA256 OpenCL]... |/-\|/-\PASS

Testing: sha512crypt-opencl, crypt(3) $6$ (rounds=5000) [SHA512 OpenCL]... |/-\|/-PASS

Testing: descrypt-opencl, traditional crypt(3) [DES OpenCL]... \|/-\|/PASS

Testing: dmg-opencl, Apple DMG [PBKDF2-SHA1 OpenCL 3DES/AES 4x]... -\|/-\|/PASS

Testing: electrum-modern-opencl, Electrum Wallet 2.8+ [SHA256 AES 32/64]... -\|PASS

Testing: encfs-opencl, EncFS [PBKDF2-SHA1 OpenCL 4x AES/Blowfish]... /-\|/PASS

Testing: enpass-opencl, Enpass Password Manager [PBKDF2-SHA1 AES OpenCL]... -\PASS

Testing: ethereum-opencl, Ethereum Wallet [PBKDF2-SHA256 OpenCL AES]... |/-PASS

Testing: ethereum-presale-opencl, Ethereum Presale Wallet [PBKDF2-SHA256 AES OpenCL]... \|/-PASS

Testing: FVDE-opencl, FileVault 2 [PBKDF2-SHA256 AES OpenCL]... \|/PASS

Testing: geli-opencl, FreeBSD GELI [PBKDF2-SHA512 OpenCL AES]... -\|PASS

Testing: iwork-opencl, Apple iWork '09 / '13 / '14 [PBKDF2-SHA1 AES OpenCL]... /-\|/PASS

Testing: keychain-opencl, Mac OS X Keychain [PBKDF2-SHA1 OpenCL 3DES]... -\|/-\PASS

Testing: keyring-opencl, GNOME Keyring [SHA256 OpenCL AES]... |/-PASS

Testing: keystore-opencl, Java KeyStore [SHA1 OpenCL]... \|/-\|/-\|/-\|/-\PASS

Testing: krb5pa-md5-opencl, Kerberos 5 AS-REQ Pre-Auth etype 23 [MD4 HMAC-MD5 RC4 OpenCL]... |/-\|/-\|/PASS

Testing: krb5pa-sha1-opencl, Kerberos 5 AS-REQ Pre-Auth etype 17/18 [PBKDF2-SHA1 OpenCL 4x]... -\|/-\|/PASS

Testing: LM-opencl [DES BS OpenCL]... -\|/-\|/-\PASS

Testing: mscash-opencl, M$ Cache Hash [MD4 OpenCL]... |/-\|/-\|/-PASS

Testing: mscash2-opencl, MS Cache Hash 2 (DCC2) [PBKDF2-SHA1 OpenCL]... \|/-\|/-\|/-\|/-\|/-\|PASS

Testing: mysql-sha1-opencl, MySQL 4.1+ [SHA1 OpenCL]... /-\|/-\|/-\|/-PASS

Testing: nt-opencl, NT [MD4 OpenCL]... \|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\PASS

Testing: ntlmv2-opencl, NTLMv2 C/R [MD4 HMAC-MD5 OpenCL]... |/-\|/-\|/-PASS

Testing: o5logon-opencl, Oracle O5LOGON protocol [SHA1 AES OpenCL]... \|/-\|PASS

Testing: ODF-opencl [SHA1 OpenCL Blowfish]... /-\|/PASS

Testing: ODF-AES-opencl [SHA256 PBKDF2-SHA1 AES OpenCL]... -\|PASS

Testing: office-opencl, MS Office [SHA1/SHA512 AES OpenCL]... /-\|/PASS

Testing: PBKDF2-HMAC-SHA256-opencl [PBKDF2-SHA256 OpenCL]... -\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: PBKDF2-HMAC-SHA512-opencl, GRUB2 / OS X 10.8+ [PBKDF2-SHA512 OpenCL]... /-\|/-\|/-\|PASS

Testing: pfx-opencl, PKCS12 PBE (.pfx, .p12) [SHA1 OpenCL]... /-\|/-PASS

Testing: phpass-opencl ($P$9 lengths 0 to 15) [MD5 OpenCL]... \|/-\|/-\|/-\|/-\PASS

Testing: pwsafe-opencl, Password Safe [SHA256 OpenCL]... |/-\|/PASS

Testing: RAKP-opencl, IPMI 2.0 RAKP (RMCP+) [HMAC-SHA1 OpenCL 4x]... -\|/-PASS

Testing: Raw-MD4-opencl [MD4 OpenCL]... \|/-\|/-\|/-\|/-PASS

Testing: Raw-MD5-opencl [MD5 OpenCL]... \|/-\|/-PASS

Testing: Raw-SHA1-opencl [SHA1 OpenCL]... \|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: Raw-SHA256-opencl [SHA256 OpenCL]... /-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|/-\|PASS

Testing: Raw-SHA512-opencl [SHA512 OpenCL]... /-\|/-\|/-\|PASS

Testing: salted-sha1-opencl [SHA1 OpenCL]... /-\|/-\|/-\|/-\|/-\|/-\|/PASS

Testing: 7z-opencl, 7-Zip (512K iterations) [SHA256 AES OPENCL]... -\|/-PASS

Testing: SL3-opencl, Nokia operator unlock [SHA1 OpenCL]... \|/PASS

Testing: strip-opencl, STRIP Password Manager [PBKDF2-SHA1 OpenCL]... -\|PASS

Testing: sxc-opencl, StarOffice .sxc [PBKDF2-SHA1 OpenCL Blowfish]... /-\|/PASS

Testing: wpapsk-opencl, WPA/WPA2 PSK [PBKDF2-SHA1 OpenCL 4x]... -\|/-PASS

Testing: XSHA512-opencl, Mac OS X 10.7 salted [SHA512 OpenCL]... \|/-\|/-\|/-\|/-\|/-PASS

Testing: zip-opencl, ZIP [PBKDF2-SHA1 OpenCL]... \|/PASS

All 60 formats passed self-tests!



travis_time:end:133983c1:start=1500023609258963000,finish=1500024424610113000,duration=815351150000
[0K

[32;1mThe command ".travis/check.sh" exited with 0.[0m



Done. Your build exited with 0.

/Users/travis/.travis/job_stages: line 161: shell_session_update: command not found

