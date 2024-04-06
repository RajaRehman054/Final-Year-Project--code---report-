import { Text, View, StyleSheet } from "react-native";
import { Avatar } from "react-native-paper";
import url from "../../utils/url.js";
import img1 from "../../assets/img1.png";

export default function AppliedCad({ data }) {
  return (
    <View style={styles.notificationbar}>
      <View style={styles.iconimg}>
        <Avatar.Image
          source={
            data.picture
              ? {
                  uri: `${url}/users/picture?path=${data.picture}`,
                }
              : img1
          }
          size={50}
        />
      </View>
      <View style={styles.notifytext}>
        <View
          style={{
            flexDirection: "row",
            justifyContent: "space-between",
            height: "100%",
            width: "100%",
            alignItems: "center",
          }}
        >
          <Text style={{ fontWeight: "bold", color: "black", marginLeft: 20 }}>
            {data.username}
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  notificationbar: {
    height: 60,
    flexDirection: "row",
    backgroundColor: "white",
    marginTop: 10,
    borderBottomWidth: 2,
    borderColor: "#d9d9d9",
    width: "100%",
    flexDirection: "row",
  },
  iconimg: {
    width: "20%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  notifytext: {
    width: "80%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
});
