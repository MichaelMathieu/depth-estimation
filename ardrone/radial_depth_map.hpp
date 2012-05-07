
// SphericCoordinates

RadialDepthMap::SphericCoordinates::SphericCoordinates(RadialDepthMap* map, float rho, float theta)
  :map(map), rho(rho), theta(theta) {
}

RadialDepthMap::CartesianCoordinates RadialDepthMap::SphericCoordinates::toCartesianCoordinates() const {
  return CartesianCoordinates(map, rho*cos(theta), rho*sin(theta));
}

// CartesianCoordinates

RadialDepthMap::CartesianCoordinates::CartesianCoordinates(RadialDepthMap* map, float x, float y)
  :map(map), x(x), y(y) {
}

void RadialDepthMap::CartesianCoordinates::add(float xToAdd, float yToAdd) {
  x += xToAdd;
  y += yToAdd;
}

RadialDepthMap::SphericCoordinates RadialDepthMap::CartesianCoordinates::toSphericCoordinates() const {
  return SphericCoordinates(map, sqrt(x*x+y*y), atan2(y, x));
}

// Ray

RadialDepthMap::Ray::Ray(RadialDepthMap* map, float theta)
  :map(map), theta(theta) {
}

// Map

// void RadialDepthMap::Map::decrement() {
//   if (nRefs) {
//     --(*nRefs);
//     if (*nRefs <= 0) {
//       delete nRefs;
//       delete map;
//     }
//   }
// }

RadialDepthMap::Map::Map()
  :map(NULL), dim(0) {
}

RadialDepthMap::Map::Map(size_t dim, float init)
  :map(new std::vector<float>(dim)),
    var(new std::vector<float>(dim)) {
  for (size_t i = 0; i < dim; ++i) {
    (*map)[i] = init;
    (*var)[i] = 1;
  }
}

RadialDepthMap::Map::Map(const Map & src)
  :map(src.map), var(src.var), dim(src.dim) {
}

RadialDepthMap::Map & RadialDepthMap::Map::operator=(const Map & src) {
  if (&src != this) {
    //decrement();
    map = src.map;
    var = src.var;
  }
  return *this;
}

RadialDepthMap::Map::~Map() {
  //delete map;
  //delete var;
}

size_t RadialDepthMap::Map::size() const {
  return map->size();
}

const float & RadialDepthMap::Map::operator() (size_t i) const {
  return (*map)[i];
}

float & RadialDepthMap::Map::operator() (size_t i) {
  return (*map)[i];
}

float RadialDepthMap::Map::getVariance(size_t i) {
  return (*var)[i];
}

void RadialDepthMap::Map::setVariance(size_t i, float variance) {
  (*var)[i] = variance;
}

RadialDepthMap::Map RadialDepthMap::Map::clone() const {
  Map ret(size());
  for (size_t i = 0; i < size(); ++i) {
    ret(i) = (*map)[i];
    //ret.setVariance(i, (*map).getVariance(i));
  }
  return ret;
} 

// RadialDepthMap

size_t RadialDepthMap::getIThetaFromTheta(float theta) const {
  //printf("sight %f\n", theta_sight);
  float theta_rectified = theta + theta_sight;
  if (theta_rectified > PI)
    theta_rectified -= 2.0f * PI;
  return floor(((theta_rectified) / (2.0f * PI) + 0.5f) * (float)nBinsTheta());
}

float RadialDepthMap::getThetaFromITheta(size_t iTheta) const {
  //return iTheta*2*PI/(float)nBinsTheta()-theta_sight;
  return getTheta1FromITheta(iTheta) + 1.0f*PI/(float)nBinsTheta();
}

float RadialDepthMap::getTheta1FromITheta(size_t iTheta) const {
  float theta = ((float)iTheta / (float)nBinsTheta() - 0.5f) * 2.0f * PI -theta_sight;
  if (theta < -PI)
    return theta + 2.0f * PI;
  else
    return theta;
}

float RadialDepthMap::getTheta2FromITheta(size_t iTheta) const {
  return getTheta1FromITheta(iTheta) + 2.0f*PI/(float)nBinsTheta();
}

size_t RadialDepthMap::nBinsTheta () const {
  return map.size();
}
