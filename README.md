# Image comparing to find duplicates

### Stuff to try
- [ ] Parallelize the code
- [x] Convert images to hash and use it to compare it other. Store the hashs in a nosql database 
      (https://github.com/JohannesBuchner/imagehash) (https://github.com/johnbumgarner/facial_similarities#imagehash-library)
- [ ] Stop comparing when at least one duplicated image is found
- [ ] Divide comparing in batchs. If a batch is not equal then stop comparing