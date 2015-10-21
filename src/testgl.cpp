#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <math.h>

using namespace std;

int main() {
	glm::mat4 trans;
	glm::vec3 nO(0.0f, 0.0f, 1.0f);
	glm::vec3 nN(1.0f, 0.0f, 0.0f);
	trans = glm::rotate(trans, glm::radians(acos(dot(nO, nN))), cross(nO, nN));
	
}