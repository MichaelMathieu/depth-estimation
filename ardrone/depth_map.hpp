

// BinIndex

DepthMap::BinIndex::BinIndex(DepthMap* map, size_t iRho, size_t iTheta)
  :map(map), iRho(iRho), iTheta(iTheta) {
}

DepthMap::SphericCoordinates DepthMap::BinIndex::toSphericCoordinates() const {
  return SphericCoordinates(map, map->getRhoFromIRho(iRho), map->getThetaFromITheta(iTheta));
}

DepthMap::CartesianCoordinates DepthMap::BinIndex::toCartesianCoordinates() const {
  return toSphericCoordinates().toCartesianCoordinates();
}

float DepthMap::BinIndex::getRho1() const {
  return map->getRho1FromIRho(iRho);
}

float DepthMap::BinIndex::getRho2() const {
  return map->getRho2FromIRho(iRho);
}

float DepthMap::BinIndex::getTheta1() const {
  return map->getTheta1FromITheta(iTheta);
}

float DepthMap::BinIndex::getTheta2() const {
  return map->getTheta2FromITheta(iTheta);
}

float & DepthMap::BinIndex::value() {
  return map->map(iTheta, iRho);
}

const float & DepthMap::BinIndex::value() const {
  return map->map(iTheta, iRho);
}

// SphericCoordinates

DepthMap::SphericCoordinates::SphericCoordinates(DepthMap* map, float rho, float theta)
  :map(map), rho(rho), theta(theta) {
}
 
DepthMap::BinIndex DepthMap::SphericCoordinates::toBinIndex() const {
  return BinIndex(map, map->getIRhoFromRho(rho), map->getIThetaFromTheta(theta));
}

DepthMap::CartesianCoordinates DepthMap::SphericCoordinates::toCartesianCoordinates() const {
  return CartesianCoordinates(map, rho*cos(theta), rho*sin(theta));
}

// CartesianCoordinates

DepthMap::CartesianCoordinates::CartesianCoordinates(DepthMap* map, float x, float y)
  :map(map), x(x), y(y) {
}

void DepthMap::CartesianCoordinates::add(float xToAdd, float yToAdd) {
  x += xToAdd;
  y += yToAdd;
}

 
DepthMap::BinIndex DepthMap::CartesianCoordinates::toBinIndex() const {
  return toSphericCoordinates().toBinIndex();
}

DepthMap::SphericCoordinates DepthMap::CartesianCoordinates::toSphericCoordinates() const {
  return SphericCoordinates(map, sqrt(x*x+y*y), atan2(y, x));
}

// BinRay

DepthMap::BinRay::BinRay(DepthMap* map, float theta)
  :map(map), theta(theta) {
}
 
size_t DepthMap::BinRay::getIBinFromDepth(float depth) const {
  return map->getIRhoFromRho(depth);
}
 
const DepthMap::BinIndex DepthMap::BinRay::operator[] (size_t i) const {
  return BinIndex(map, i, map->getIThetaFromTheta(theta));
}

DepthMap::BinIndex DepthMap::BinRay::operator[] (size_t i) {
  return BinIndex(map, i, map->getIThetaFromTheta(theta));
};

// Map

void DepthMap::Map::decrement() {
  if (nRefs) {
    --(*nRefs);
    if (*nRefs <= 0) {
      delete nRefs;
      delete map;
    }
  }
}

DepthMap::Map::Map()
  :nRefs(NULL), map(NULL), d2(0) {
}

DepthMap::Map::Map(size_t d1, size_t d2, float init)
  :nRefs(new int(1)), map(new std::vector<std::vector<float> >(d1)), d2(d2) {
  for (size_t i = 0; i < d1; ++i)
    (*map)[i] = std::vector<float>(d2, init);
}

DepthMap::Map::Map(const Map & src)
  :nRefs(src.nRefs), map(src.map), d2(src.d2) {
  ++(*nRefs);
}

DepthMap::Map & DepthMap::Map::operator=(const Map & src) {
  if (&src != this) {
    decrement();
    nRefs = src.nRefs;
    map = src.map;
    if (nRefs)
      ++(*nRefs);
  }
  return *this;
}

DepthMap::Map::~Map() {
  decrement();
}

size_t DepthMap::Map::size1() const {
  return map->size();
}

size_t DepthMap::Map::size2() const {
  return d2;
}

const float & DepthMap::Map::operator() (size_t i, size_t j) const {
  return (*map)[i][j];
}

float & DepthMap::Map::operator() (size_t i, size_t j) {
  return (*map)[i][j];
}

DepthMap::Map DepthMap::Map::clone() const {
  Map ret(size1(), size2());
  for (size_t i = 0; i < size1(); ++i)
    for (size_t j = 0; j < size2(); ++j)
      ret(i, j) = (*map)[i][j];
  return ret;
} 

// DepthMap

size_t DepthMap::getIRhoFromRho(float rho) const {
  if (rho > maxDepth)
    return nBinsRho()-1;
  else
    return floor(rho / maxDepth * (float)nBinsRho());
}

size_t DepthMap::getIThetaFromTheta(float theta) const {
  float theta_rectified = theta + theta_sight;
  if (theta_rectified > PI)
    theta_rectified -= 2.0f * PI;
  return floor(((theta_rectified) / (2.0f * PI) + 0.5f) * (float)nBinsTheta());
}

float DepthMap::getRhoFromIRho(size_t iRho) const {
  return ((float)iRho + 0.5f) * maxDepth/(float)nBinsRho();
}

float DepthMap::getRho1FromIRho(size_t iRho) const {
  return (float)iRho * maxDepth/(float)nBinsRho();
}

float DepthMap::getRho2FromIRho(size_t iRho) const {
  return ((float)iRho + 1.0f) * maxDepth/(float)nBinsRho();
}

float DepthMap::getThetaFromITheta(size_t iTheta) const {
  return getTheta1FromITheta(iTheta) + 1.0f*PI/(float)nBinsTheta();
  /*
  float theta = (((float)iTheta + 0.5f) / (float)nBinsTheta() - 0.5f) * 2.0f * PI -theta_sight;
  if (theta < -PI)
    return theta + 2.0f * PI;
  else
  return theta;*/
}

float DepthMap::getTheta1FromITheta(size_t iTheta) const {
  float theta = ((float)iTheta / (float)nBinsTheta() - 0.5f) * 2.0f * PI -theta_sight;
  if (theta < -PI)
    return theta + 2.0f * PI;
  else
    return theta;
}

float DepthMap::getTheta2FromITheta(size_t iTheta) const {
  return getTheta1FromITheta(iTheta) + 2.0f*PI/(float)nBinsTheta();
}

size_t DepthMap::nBinsRho () const {
  return map.size2();
}

size_t DepthMap::nBinsTheta () const {
  return map.size1();
}
