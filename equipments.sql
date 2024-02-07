/*
 Navicat MySQL Data Transfer

 Source Server         : llm_test
 Source Server Type    : MySQL
 Source Server Version : 80300
 Source Host           : localhost:3306
 Source Schema         : equipments

 Target Server Type    : MySQL
 Target Server Version : 80300
 File Encoding         : 65001

 Date: 04/02/2024 17:33:55
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for equipments
-- ----------------------------
DROP TABLE IF EXISTS `equipments`;
CREATE TABLE `equipments`  (
  `id` int(0) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `type` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `equipment` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `create_time` datetime(0) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of equipments
-- ----------------------------
INSERT INTO `equipments` VALUES (1, '中碎筛分除尘器差压超过1500Pa', '差压过高', '中碎筛分除尘器', '2023-09-08 21:07:23');
INSERT INTO `equipments` VALUES (2, '中碎筛分除尘器差压超过1500Pa', '差压过高', '中碎筛分除尘器', '2023-10-13 12:32:05');
INSERT INTO `equipments` VALUES (3, '中碎筛分除尘器输灰不畅', '输灰不畅', '中碎筛分除尘器', '2023-12-01 02:15:38');
INSERT INTO `equipments` VALUES (4, '粗碎除尘器差压过高', '差压过高', '粗碎除尘器', '2024-01-18 17:20:37');

SET FOREIGN_KEY_CHECKS = 1;
