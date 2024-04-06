import React from "react";
import { IoCheckmarkCircle } from "react-icons/io5";
import { HOSTNAME } from "../../../Config";
import UserOne from "../../assets/img1.png";
import { format } from "timeago.js";
import { useSelector } from "react-redux";

const PaymentHistoryTable = (props) => {
  const { user } = useSelector((state) => state.user);
  const data = props.data;

  return (
    <table className="w-full table-auto rounded-xl bg-white text-[12px] font-normal text-text shadow-xl md:text-[14px]">
      <thead>
        <tr className="border-b-[3px] border-[#E4E4E4] ">
          <th className="pb-[8px] pl-4 pt-[18px] text-left">User</th>
          <th className="pb-[8px] pt-[18px] text-left">Amount Added</th>
          <th className="pb-[8px] pt-[18px] text-left">Time</th>
          <th className="pb-[8px] pt-[18px] text-left">Actions</th>
        </tr>
      </thead>
      <tbody>
        {data.map((element, index) => (
          <tr
            key={index}
            className="cursor-pointer rounded-xl border-b-[1px] border-[#E4E4E4] bg-white hover:bg-[#EBEBEB]"
          >
            <td className="pl-3">
              <div className="py-2 ">
                <div className="flex h-[50px] w-[50px] items-center justify-center overflow-hidden rounded-full">
                  <img
                    src={
                      user.picture !== null
                        ? `${HOSTNAME}users/picture?path=${user.picture}`
                        : UserOne
                    }
                  />
                </div>
              </div>
            </td>
            <td className="font-medium">
              <p>{element.amount}</p>
            </td>
            <td className="font-medium">
              <p className="pr-1">{format(element.createdAt)}</p>
            </td>

            <td>
              <div className="justify-center">
                <IoCheckmarkCircle
                  size={24}
                  color="green"
                  onClick={() => alert("Transaction made")}
                />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default PaymentHistoryTable;
